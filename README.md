# UTKFace Multitask Age Classification

This project implements a multitask deep learning framework for age classification using the UTKFace dataset. It supports both **contrastive pretraining** and **classification-only** modes.

## 🧠 Features
- Age classification into 6 age groups
- Optional binary face segmentation as an auxiliary task
- Contrastive pretraining using InfoNCE
- Support for different learning rate strategies (uniform, differentiated)
- ResNet18 backbone with custom segmentation decoder
- Built from scratch using PyTorch and Albumentations

---

## 🔧 Environment Setup

We recommend using **Python 3.10** in a virtual environment. You may choose between `venv` or `conda`.

### Option 1: Using `venv`
```bash
python3.10 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .
```

### Option 2: Using Anaconda
```bash
conda create -n utkface python=3.10 -y
conda activate utkface
pip install -r requirements.txt
```

---

## 🚀 Running the Project

Once the environment is ready, run the training pipeline using:

### For Contrastive Pretraining
```bash
python -m utkface_multitask  contrastive
```

### For Classification (or Multitask Fine-Tuning)
```bash
python -m utkface_multitask classification
```

Additional arguments (like learning rate, epochs, multitask flag, etc.) can be set in `configs/contrastive.yaml` or `configs.multitask.yaml`

---

## 📁 Dataset Preparation

Before running training or evaluation, **you must first generate face masks** and then **split the dataset**.

1. **Run inference to generate segmentation masks**:
   ```bash
   python utkface_multitask/mask_inference/inference.py
   ```

2. **Split the dataset afterwards.**  
   The final dataset folder should be structured as follows:

```
dataset/
└── processed_Split/
    ├── train/
    │   ├── 000001_image.png
    │   ├── 000001_mask.png
    │   └── ...
    ├── val/
    │   ├── 000101_image.png
    │   ├── 000101_mask.png
    │   └── ...
    └── test/
        ├── 000201_image.png
        ├── 000201_mask.png
        └── ...
```

Each image should have a corresponding binary mask with matching filenames, where `_image.png` and `_mask.png` pairs are used for both classification and segmentation tasks.

---

## 📊 Results

- Highest accuracy: **74.0%**
- Best model: **Multitask + Contrastive Pretraining + Differentiated LR**

### Experimental Summary
This result highlights the importance of combining both self-supervised representation learning and auxiliary supervision (via segmentation). Using a lower learning rate for pretrained encoders while keeping task heads with standard LR provided the best generalization.

### Summary Table
| LR Strategy | Task        | Pretrained Weights | Accuracy | AUC  | F1   | Precision |
|-------------|-------------|---------------------|----------|------|------|-----------|
| Uniform     | Classification | Contrastive      | 73.0%    | 93.0 | 70.0 | 76.3      |
| Uniform     | Classification | ImageNet         | 72.0%    | 93.3 | 72.0 | 76.0      |
| Uniform     | Multitask      | Contrastive      | 73.5%    | 93.0 | 73.5 | 75.8      |
| Uniform     | Multitask      | ImageNet         | 72.3%    | 93.3 | 72.3 | 76.4      |
| Differentiated | Multitask  | Contrastive       | **74.0%** | **94.0** | **74.1** | **78.2** |
| Differentiated | Multitask  | ImageNet          | 70.0%    | 92.0 | 69.5 | 71.5      |

### Confusion Matrix

```markdown
![Confusion Matrix](utkface_multitask/figures/confusion_matrix.png)
```
This matrix helps interpret how well the model distinguishes among different age groups.

---

## 📜 License

MIT License © 2025 Umut Dündar
