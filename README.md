# DINOv2_Mod

A custom Python package for image feature extraction, logistic regression training, and prediction using DINOv2.

## Folder Structure
- **dinov2_mod/func**: Custom functions for the pipeline.
- **dinov2_mod/pretrained_weights**: Pretrained weights for DINOv2.
- **dinov2_mod/supplementary**: Supporting files like `label_map.json`.logistic_regression_model.
- **dinov2_mod/tests**: Test scripts.

## Installation
Clone the Repository

```bash
git clone https://github.com/Raj-codes/dinov2_mod.git
cd dinov2_mod
```

Set Up a Python Environment

```bash
conda create -n dinov2mod python=3.9 -y
conda activate dinov2mod
```

Install Dependencies

A. To install all dependencies for training and evaluation(expects Linux environment):

```bash
pip install -r requirements.txt
```

B. To only run the pretrained model

```bash
pip install -r requirements_eval_windows.txt
```

Install the dinov2_mod package using setup.py

```bash
pip install -e .
```