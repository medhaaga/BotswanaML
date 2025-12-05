# BotswanaML


## Reproducing Results

Open a terminal and run:

```bash
git clone https://github.com/medhaaga/BotswanaML.git
cd <BotswanaML>
```
This project is built using **Python 3.11+**.

Download Python from here [here](https://www.python.org/downloads/). To check your version:

```bash
python3 --version
```

If Python is unavailable, install it using the installer for your operating system.

Next, install Miniconda. Miniconda is a minimal Python distribution that makes environment management simple.
Download Miniconda [here](https://docs.conda.io/en/latest/miniconda.html)

After installation, restart your terminal and verify:

```bash
conda --version
```

Create a conda environment:
```bash
conda create -n wildlife python=3.11 numpy scipy pandas scikit-learn matplotlib seaborn ipython jupyterlab -c conda-forge 
```

Then activate the environment:

```bash
conda activate wildlife
```
Install pip dependencies:
```bash
pip install pot==0.9.5 tqdm
```
To confirm:

```bash
conda env list
```

PyTorch is **not** included in the environment because installation depends on your hardware (CPU-only, CUDA, ROCm, etc.). Visit the official PyTorch installation selector [here](https://pytorch.org/get-started/locally/).

Examples:

**CPU-only**

```bash
pip install torch torchvision torchaudio
```

**CUDA 12.4**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
To verify successful torch installation, run: 

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```
