
# BotswanaML

This repository contains code for preprocessing wildlife sensor data and training domain adaptation models to transfer behavior prediction from high-frequency Vectronics acceleration data to RVC data.

---

## Environment Setup

Following is the environment setup for **macOS or Linux** users.
If you are using **Windows**, first set up a Linux environment using **WSL** by following the instructions in [Windows Users (WSL Setup)](#windows-users-wsl-setup), then return to this section and follow the steps below **inside the WSL terminal**.

---

### Step 1: Clone the repository

Open a terminal and run:

```bash
git clone https://github.com/medhaaga/BotswanaML.git
cd BotswanaML
```

---

### Step 2: Install Python

This project requires **Python 3.11 or higher**. Download Python from the official website:
[https://www.python.org/downloads/](https://www.python.org/downloads/)

To check your Python version:

```bash
python3 --version
```

---

### Step 4: Install Miniconda

Miniconda is a lightweight Python distribution that simplifies dependency and environment management.

Download Miniconda from:
[https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

After installation, restart your terminal and verify:

```bash
conda --version
```

---

### Step 5: Create and activate the environment

Create a new conda environment named `wildlife`:

```bash
conda create -n wildlife python=3.11 numpy scipy pandas scikit-learn matplotlib seaborn ipython jupyterlab PyYAML -c conda-forge
```

Activate the environment:

```bash
conda activate wildlife
```

Install additional dependencies using pip:

```bash
pip install pot==0.9.5 tqdm
```

Verify that the environment was created successfully:

```bash
conda env list
```
Whenever running python scripts from this repository, activate the environment using the command above. Whenever running a Jupyter notebook from this repository, choose the Python kernel "wildlife" from the available list.
---

### Step 6: Install PyTorch (hardware-specific)

PyTorch is **not included** in the environment because installation depends on your hardware configuration (CPU-only, CUDA, ROCm, etc.).

Use the official PyTorch installation selector:
[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

Examples:

**CPU-only**

```bash
pip install torch torchvision torchaudio
```

**CUDA 12.1 (conda)**

```bash
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

**CUDA 12.4 (pip)**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verify the installation:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

---

### Windows Users (WSL Setup)

If you are using **Windows**, you must first set up **Windows Subsystem for Linux (WSL2)**.

1. Open **PowerShell as Administrator**
2. Run:

   ```powershell
   wsl --install
   ```
3. Restart your computer when prompted
4. Open **Ubuntu** from the Start menu and complete the initial setup

Once Ubuntu is installed:

* Use the **Ubuntu terminal** (not Windows Command Prompt)
* Follow all steps in the **Environment Setup** section above inside WSL

Official WSL documentation:
[https://learn.microsoft.com/en-us/windows/wsl/](https://learn.microsoft.com/en-us/windows/wsl/)

---

## Overall Project Workflow 

```text
Raw Sensor Data
│
├── Vectronics (16 Hz acceleration, per individual/year)
│
└── RVC (30-second summaries, all individuals)
        │
        ▼
Data Preparation & Preprocessing
│
├── Vectronics preprocessing
│     ├── segmentation
│     ├── metadata creation
│     ├── annotation matching
│     └── feature extraction (30s windows)
│
└── RVC preprocessing
      ├── XML parsing
      ├── calibration
      └── thresholding
        │
        ▼
Model Training (Domain Adaptation)
│
├── CORAL (unsupervised domain adaptation)
└── FixMatch (semi-supervised domain adaptation)
        │
        ▼
Model Calibration & Evaluation
```

The *Vectronics is the source domain* and *RVC is the target domain*.

Here is a **very concise**, minimally worded revision that preserves your original structure and format while correcting grammar and improving clarity:

---

## Data Preparation

This repository uses two primary data sources:

1. **Vectronics data**
   High-frequency tri-axial acceleration data collected at **16 Hz**, stored as raw CSV files per individual and year.

2. **RVC data**
   Coarse-grained acceleration summary statistics collected at **30-second intervals**, stored as a single CSV file.

The following path updates and data placement steps are required before running any preprocessing or training code:

1. `VECTRONICS_PATHS` in `config/paths.py` must point to the directories containing raw Vectronics data for each individual.
   The expected directory structure is described in [Vectronics Data](#vectronics-data).

2. `RVC_ACC_ANNOTATED` in `config/paths.py` must point to the consolidated RVC acceleration CSV file.
   The expected file format is described in [RVC Data](#rvc-data).

3. Audio and video annotation CSV files are expected at:

   * `data/2025_10_31_awd_audio_annotations.csv`
   * `data/2025_10_31_awd_video_annotations.csv`

   *If your filenames differ, update `get_audio_labels_path` and `get_video_labels_path` in `src/utils/io.py`.*

4. RVC deployment header files (XML) must be placed in `data/RVC_header_files/`.

5. RVC metadata must be placed in `data/RVC_metadata.xlsx`.

---


## Vectronics Data

### Directory structure

The Vectronics data should be organized as follows:

```
<Vectronics data dir>/
├── <individual_1>/
│   ├── 2022.csv
│   ├── 2023.csv
│   └── 2024.csv
├── <individual_2>/
│   ├── 2022.csv
│   └── 2023.csv
└── <individual_3>/
    ├── 2023.csv
    └── 2024.csv
    ...
```

At the time of repository creation, data are available for the following individuals:
**jessie**, **green_palus**, **ash**, **fossey**

---

### Configuring data paths

Paths to each individual’s raw Vectronics data directory are defined in:

```
config/paths.py
```

within the dictionary `VECTRONICS_PATHS`.

Ensure that:

* dictionary keys (individual names) are **lowercase**
* dictionary values point to directories containing year-wise CSV files

Modify this dictionary to match your local data locations.

---

### Vectronics preprocessing pipeline

The conversion of raw Vectronics data into coarse summaries aligned with the RVC temporal scale involves multiple steps.

Implementation files:

* `src/utils/vectronics_data_prep.py`
* `src/utils/vectronics_preprocessing.py`

A step-by-step walkthrough is provided in:

```
notebooks/Vectronics_preprocessing.ipynb
```

---

#### Step 1: Raw data preparation

Implemented in `src/utils/vectronics_data_prep.py`:

|                                   Step | Function                             |
| -------------------------------------: | ------------------------------------ |
|               Create half-day segments | `create_vectronics_halfday_segments` |
|                        Create metadata | `create_metadata`                    |
|    Combine video and audio annotations | `combined_annotations`               |
| Match acceleration and annotation data | `create_matched_data`                |

Run all steps using:

```bash
conda activate wildlife
python vectronics_data_prep.py
```

This script takes approximately **3 hours** to run and only needs to be executed **once**.

---

#### Step 2: Feature generation and windowing

Implemented in `src/utils/vectronics_preprocessing.py`:

|                                       Step | Function                   |
| -----------------------------------------: | -------------------------- |
|           Create 30-second labeled windows | `create_max_windows`       |
|                 Compute summary statistics | `create_summary_data`      |
| Window unlabeled data and compute features | `create_windowed_features` |

Run the preprocessing pipeline using:

```bash
python scripts/run_Vectronics_preprocessing.py
```

Alternatively, these steps can be explored interactively in:

```
notebooks/Vectronics_preprocessing.ipynb
```

---

## RVC Data

### Required input files

To create the RVC dataset, the following files are required:

1. A single CSV file containing RVC acceleration data for all individuals
   (path defined as `RVC_ACC_ANNOTATED` in `config/paths.py`)
2. An RVC metadata file containing sensor deployment information:
   `data/RVC_metadata.xlsx`
3. RVC header XML files containing sensor configuration details:
   `data/RVC_header_files/*`

*Expected formats and column names are documented in `notebooks/RVC_preprocessing.ipynb`.*

Create the processed RVC dataset by running:

```bash
python scripts/run_RVC_preprocessing.py
```

Alternatively, follow the notebook walkthrough:

```
notebooks/RVC_preprocessing.ipynb
```

---

### RVC preprocessing functions

Implemented in `src/utils/RVC_preprocessing.py`:

|               Step | Function             |
| -----------------: | -------------------- |
|    Parse XML files | `parse_xml_file`     |
| Calibrate RVC data | `calibrate_RVC_data` |
| Apply thresholding | `threshold_RVC`      |

---

## Model Training

This repository implements two domain adaptation approaches to transfer behavior prediction from **Vectronics (source)** to **RVC (target)** data:

* **CORAL** (unsupervised; no weak labels)
* **FixMatch** (semi-supervised; uses weak labels)

The RVC target domain is split into two subdomains corresponding to firmware versions **v2** and **v3**, which exhibit substantial distribution shift. Separate models are trained for each subdomain.

---

### CORAL Model

Implementation:

* Model: `src/methods/coral.py`
* Training logic: `train_coral` in `src/utils/train.py`

Train a CORAL model using:

```bash
python scripts/run_coral.py --exp_name <value> --pos_idx <value> --center_idx <value>
```

Results are saved to:

```
results/domain_adaptation_training_results/coral/<exp_name>_<timestamp>/
```

Each run contains:

* `config.json`
* `target1/` and `target2/` directories with:

  * `model.pt`
  * `val_results.npz`
  * `test_results.npz`
  * `training_stats.json`

---

### FixMatch Model

Training is implemented in `train_fixmatch` within `src/utils/train.py`.

Run FixMatch using:

```bash
python scripts/run_fixmatch.py --exp_name <value> --pos_idx <value> --center_idx <value>
```

Results are saved to:

* `fixmatch_semi_supervised/` if `lambda_target > 0`
* `fixmatch_self_supervised/` otherwise

Each run saves:

* `config.json`
* `target_splits.csv`
* `target1/` and `target2/` directories with model outputs

---

## Model Calibration

Model calibration code is implemented in:

```
src/methods/calibration.py
```

Calibration and visualization for CORAL and FixMatch models are provided in:

* `notebooks/CORAL_model.ipynb`
* `notebooks/FixMatch_model.ipynb`
Below are **inline diagrams and flow summaries** you can paste directly into your README. They are designed for **absolute beginners**, use simple ASCII diagrams (GitHub-safe), and clarify *what runs when and why* without adding conceptual overload.
