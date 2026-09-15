# MSE420 AI Forensics

Course workspace for exploring trust, reliability, and accountability in AI systems.

## Repository layout

- `demonstrations/` - Instructor-led, in-class notebook demonstrations.
- `labs/` - Student lab notebooks and supporting lab materials.

## Getting started

1. Create the Conda environment from the repository root:

   ```bash
   conda env create -f environment.yml
   ```

2. Activate it and register its Jupyter kernel:

   ```bash
   conda activate mse420-ai-forensics
   python -m ipykernel install --user --name mse420-ai-forensics --display-name "MSE420 AI Forensics (Python 3.11)"
   ```

3. Start JupyterLab and select **MSE420 AI Forensics (Python 3.11)** as the notebook kernel:

   ```bash
   jupyter lab
   ```

The environment file pins the notebook dependencies, including NumPy 1.26.4, to avoid the NumPy 2 compatibility errors seen with older compiled libraries. Restart the notebook kernel after changing installed packages.

Open the repository in VS Code with the Jupyter extension or launch JupyterLab from the repository root. Keep generated checkpoints, virtual environments, and local data out of commits; the repository `.gitignore` handles the common cases.
