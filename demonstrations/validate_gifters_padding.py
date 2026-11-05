"""Offline padding-input/cache checks; uses the local tokenizer, no model download."""
import ast,nbformat,torch,json,hashlib
from types import SimpleNamespace
from transformers import AutoTokenizer
n=nbformat.read('gemma_gifters_local.ipynb',4)
ns={'cfg':SimpleNamespace(device='cpu',max_new_tokens=256,temperature=.7),'digest':lambda x:hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()}
tok=AutoTokenizer.from_pretrained('models/Qwen3-0.6B',local_files_only=True)
ns['backend']=SimpleNamespace(torch=torch,tokenizer=tok,model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=40960)))
ns['request_identity']=lambda prompt,seed,temp:{'prompt':prompt,'seed':seed,'temperature':temp,'engine':'fixture'}
for c in n.cells:
 if c.cell_type=='code':
  for node in ast.parse(c.source).body:
   if isinstance(node,ast.FunctionDef) and node.name in {'prepare_padded_inputs','padding_identity'}:
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<notebook>','exec'),ns)
build=ns['prepare_padded_inputs']; prompt='Explain how a telescope works.'
a=build(prompt,0,'masked')
rendered=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
assert torch.equal(a['input_ids'],tok(rendered,return_tensors='pt',add_special_tokens=False).input_ids)
for k in [1,2,4,8,16,32,128]:
 m=build(prompt,k,'masked');u=build(prompt,k,'attended')
 assert torch.equal(m['input_ids'],u['input_ids'])
 assert torch.equal(m['input_ids'][:,k:],a['input_ids'])
 assert bool((m['input_ids'][:,:k]==tok.pad_token_id).all())
 assert int(m['attention_mask'][:,:k].sum())==0 and int(m['attention_mask'].sum())==a['input_ids'].numel()
 assert bool((u['attention_mask']==1).all())
 assert torch.equal(m['position_ids'][:,k:],a['position_ids'])
 assert torch.equal(u['position_ids'][:,k:],a['position_ids']+k)
 assert m['input_ids'][0,-1]==a['input_ids'][0,-1]  # no right padding
ident=ns['padding_identity']
assert ident(prompt,1,0,'masked')==ident(prompt,1,0,'attended')
assert ident(prompt,1,8,'masked')!=ident(prompt,1,8,'attended')
assert ident(prompt,1,8,'masked')!=ident(prompt,1,16,'masked')
try:build(prompt,-1,'masked')
except ValueError:pass
else:raise AssertionError('negative pad count accepted')
ns['backend'].model.config.max_position_embeddings=10
try:build(prompt,128,'masked')
except ValueError:pass
else:raise AssertionError('context overflow accepted')
print('PASS: actual tokenizer pad IDs; left insertion; preserved chat suffix; masks; real-token positions; zero-padding equivalence; cache condition separation; context guard.')
# An unusable output must be checkpointed as a failure and reused, not retried.
import tempfile,time
from pathlib import Path
for c in n.cells:
 if c.cell_type=='code':
  for node in ast.parse(c.source).body:
   if isinstance(node,ast.FunctionDef) and node.name=='cached_padding_generate':
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<notebook>','exec'),ns)
ns.update(json=json,time=time,budget_check=lambda:None,cache_stats={'generation_hits':0,'generation_misses':0},generation_events=[],
          encoder=SimpleNamespace(tokenizer=lambda text,**kw:{'input_ids':[]},max_seq_length=256),
          atomic_json=lambda p,v:Path(p).write_text(json.dumps(v)))
ns['backend'].model.config.max_position_embeddings=40960
ns['backend'].model.generation_config=SimpleNamespace(eos_token_id=[tok.eos_token_id])
calls=[]
def empty_generation(**kw):
 calls.append(1)
 return torch.cat([kw['input_ids'],torch.tensor([[tok.eos_token_id]])],dim=1)
ns['backend'].model.generate=empty_generation
with tempfile.TemporaryDirectory() as tmp:
 ns['CACHE']=ns['OUT']=Path(tmp)
 a=ns['cached_padding_generate'](prompt,123,8,'attended')
 b=ns['cached_padding_generate'](prompt,123,8,'attended')
 assert a['issues']==['empty'] and a['text']=='' and a==b and len(calls)==1
 assert ns['cache_stats']=={'generation_hits':1,'generation_misses':1}
print('PASS: empty response retained as an explicit failure, cached and not resampled.')
