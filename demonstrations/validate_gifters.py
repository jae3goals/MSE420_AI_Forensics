"""Offline mechanics tests; real Qwen execution is recorded separately in the notebook."""
import ast,json,hashlib,re,difflib,tempfile,time
from pathlib import Path
from contextlib import contextmanager
from types import SimpleNamespace
import nbformat,numpy as np,pandas as pd,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

nb=nbformat.read(Path(__file__).with_name('gemma_gifters_local.ipynb'),4); nbformat.validate(nb)
names={'cosine_distance','WeightPlane','TransformersBackend','effect_stats','comparison_plot',
       'deterministic_edit','make_variants','paraphrase_rejection','atomic_json','digest','seed_for','request_identity',
       'cached_generate','response_seeds','baseline','ensemble','encode'}
ns=dict(globals()); ns['display']=lambda *a:None
for c in nb.cells:
 if c.cell_type=='code':
  tree=ast.parse(c.source)
  for node in tree.body:
   if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names:
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<notebook>','exec'),ns)
cfg=SimpleNamespace(seed=420,ensemble_size=8,baseline_pairs=20,temperature=.7,
    max_new_tokens=256,variants_per_type=2,magnitudes=(.15,.3),paraphrase_attempts=2,
    perturbation_types=('spelling','punctuation','formatting','reorder'),response_instruction='Short answer',
    max_document_tokens=192,device='cpu')
ns['cfg']=cfg
cos=ns['cosine_distance']; assert cos([1,0],[0,1])==1
try: cos([0,0],[1,0])
except ValueError: pass
else: raise AssertionError('zero centroid')
st=ns['effect_stats'](.5,np.array([.1,.2,.3,.4]))
assert st['baseline_percentile']==100 and st['upper_tail_fraction']==.2 and st['exceeds_q95']
assert ns['effect_stats'](0,np.zeros(4))['median_ratio'] is None

# Generic edits, including numbers and short/no-sentence prompts.
edit=ns['deterministic_edit']; prompt='Compare morning rainfall with evening rainfall. Explain uncertainty in the measurement.'
for method in cfg.perturbation_types:
 a=edit(prompt,method,.3,123); assert a and a!=prompt and a==edit(prompt,method,.3,123)
assert edit('123','spelling',.3,1) is None
assert edit('One sentence','reorder',.3,1) is None
assert edit('Dose 1.5 mg. Check results.', 'punctuation',.2,5).find('1.5')>=0
rows,attempts,shortfalls=ns['make_variants']([prompt,'Why?'])
for original,g in rows.groupby('original'):
 assert not (g.edited==original).any() and g.edited.is_unique
assert any(x['method']=='reorder' and x['accepted']==1 for x in shortfalls)
assert any(x['decision'] in {'inapplicable','unchanged_or_duplicate'} for x in attempts)
# Unchanged Qwen candidates must be rejected, never replaced with canned content.
old=cfg.perturbation_types; cfg.perturbation_types=('paraphrase',)
ns['cached_generate_original']=ns['cached_generate']; ns['cached_generate']=lambda *a,**k:prompt
r,a,s=ns['make_variants']([prompt]); assert r.empty and s[0]['accepted']==0
ns['cached_generate']=ns.pop('cached_generate_original'); cfg.perturbation_types=old

# Unique independent seeds and a separate diagnostic center.
all_seeds=[]
for i in range(cfg.baseline_pairs):
 for side in ['a','b']: all_seeds+=ns['response_seeds'](prompt,f'baseline/{i}/{side}')
all_seeds+=ns['response_seeds'](prompt,'anchor')+ns['response_seeds'](prompt,'plane/center')
assert len(all_seeds)==len(set(all_seeds))

model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Linear(4,2))
original={n:p.detach().clone() for n,p in model.named_parameters()}
plane=ns['WeightPlane'](model,420,1000)
for i,(name,p) in enumerate(model.named_parameters()):
 d=plane.direction(original[name],i,0)
 assert torch.allclose(d.norm(),original[name].norm(),atol=1e-6)
 assert torch.equal(d,plane.direction(original[name],i,0))
try:
 with plane.at(.01,-.02):
  assert any(not torch.equal(p,original[n]) for n,p in model.named_parameters())
  raise RuntimeError('Injected interruption')
except RuntimeError as e: assert str(e)=='Injected interruption'
assert all(torch.equal(p,original[n]) for n,p in model.named_parameters())
try: ns['WeightPlane'](model,420,1)
except ValueError: pass
else: raise AssertionError('memory guard')

# Known probabilities catch teacher-forcing off-by-one errors without changing that experiment.
class Tokenizer:
 def __call__(self,*a,**k): return SimpleNamespace(input_ids=torch.tensor([[0,1,2]]))
 def convert_ids_to_tokens(self,ids): return list(map(str,ids))
class FixedLogits:
 def __call__(self,**k): return SimpleNamespace(logits=torch.log(torch.tensor([[[.1,.8,.1],[.1,.2,.7],[.6,.2,.2]]])))
b=ns['TransformersBackend'].__new__(ns['TransformersBackend']); b.torch=torch;b.tokenizer=Tokenizer();b.model=FixedLogits()
tokens,losses=b.token_losses('fixture'); assert tokens==['1','2'];np.testing.assert_allclose(losses,-np.log([.8,.7]),rtol=1e-6)

class FakeBackend:
 def __init__(self):self.calls=0;self.fail=False
 def generate(self,prompt,seed,temperature=None):
  self.calls+=1
  if self.fail:raise RuntimeError('interrupted')
  return f'fixture response {self.calls}'
with tempfile.TemporaryDirectory() as tmp:
 ns.update(CACHE=Path(tmp),OUT=Path(tmp),ENGINE_ID='fixture',weight_state='original',
   generation_events=[],cache_stats={'generation_hits':0,'generation_misses':0},budget_check=lambda:None,backend=FakeBackend())
 gen=ns['cached_generate']; b=ns['backend']
 first=gen('hello',123); assert first==gen('hello',123) and b.calls==1
 gen('hello',123,temperature=.8); assert b.calls==2
 ns['weight_state']='perturbed'; gen('hello',123); assert b.calls==3
 ns['weight_state']='original';ns['ENGINE_ID']='new-model';gen('hello',123);assert b.calls==4
 # Failed sample leaves no valid checkpoint; resume retries only the missing response.
 b.fail=True
 try:gen('interruption',123)
 except RuntimeError:pass
 assert len(list(Path(tmp).glob('*.response.json')))==4
 b.fail=False;gen('interruption',123);calls=b.calls;gen('interruption',123);assert b.calls==calls
 frame=pd.DataFrame({'label':['A','B'],'distance':[.01,.2]})
 ns['comparison_plot'](np.array([.01,.02,.03,.04]),frame,'test.png')
 assert (Path(tmp)/'test.png').stat().st_size>1000
print('PASS: generic unique edits/shortfalls, rejection without canned fallback, independent baseline seeds, empirical comparisons, interruption/resume and cache invalidation, teacher forcing, normalized directions, exact restoration after failure, and comparison plot.')
assert ns['paraphrase_rejection']('Describe rainfall.','Rainfall is falling water.')=='request_changed_to_statement'
assert ns['paraphrase_rejection']('Describe rainfall.','Explain how rainfall occurs.') is None
assert ns['paraphrase_rejection']('What changes?','Changes happen.')=='question_changed_to_statement'
class EncoderFixture:
 max_seq_length=4
 def __init__(self): self.calls=0
 def tokenizer(self,t,truncation=False):return {'input_ids':t.split()}
 def encode(self,texts,**kwargs):
  self.calls+=1;return np.tile([1.,0.,0.],(len(texts),1))
with tempfile.TemporaryDirectory() as tmp:
 ns.update(CACHE=Path(tmp),ENCODER_ID='encoder-a',encoder=EncoderFixture(),versions={'sentence-transformers':'test'},cache_stats={'embedding_hits':0})
 ns['encode'](['short text']);ns['encode'](['short text']);assert ns['encoder'].calls==1
 ns['ENCODER_ID']='encoder-b';ns['encode'](['short text']);assert ns['encoder'].calls==2
 try:ns['encode'](['one two three four five'])
 except ValueError:pass
 else:raise AssertionError('silent encoder truncation')
print('PASS: request-form rejection, encoder cache identity, and no silent embedding truncation.')
# Fresh and resumed encodings must use identical precision as well as values.
with tempfile.TemporaryDirectory() as tmp:
 ns.update(CACHE=Path(tmp),ENCODER_ID='precision-check',encoder=EncoderFixture(),versions={'sentence-transformers':'test'},cache_stats={'embedding_hits':0})
 first=ns['encode'](['short text']);second=ns['encode'](['short text'])
 assert first.dtype==second.dtype==np.float32
 np.testing.assert_array_equal(first,second)
print('PASS: fresh/checkpoint embedding precision and values match exactly.')
