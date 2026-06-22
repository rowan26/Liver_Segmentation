# 🫁 Liver Segmentation using U-Net and MONAI
> 3D Medical Image Segmentation of the liver from CT scans using a U-Net architecture built with MONAI and PyTorch.

---

## 📋 Project Overview

This project implements an **end-to-end pipeline** for **automatic liver segmentation** on 3D CT scans from the [Medical Segmentation Decathlon dataset](http://medicaldecathlon.com/). The goal is to produce a **binary mask** predicting, pixel by pixel, whether each voxel belongs to the liver or not.

### The full pipeline covers:
- 📂 DICOM to NIfTI conversion
- 🔪 Data preparation (grouping into 75-slice chunks)
- ⚙️ Preprocessing (resizing, intensity normalization)
- 🧠 U-Net training with Dice Loss + Data Augmentation
- 📊 Evaluation and visualization
- 🧹 Post-processing (connected component filtering)
- ☁️ GPU training on Google Colab (Tesla T4)

---

## 📊 Results

| Metric | Train | Test (raw) | Test (post-processed) |
|--------|:-----:|:----------:|:---------------------:|
| **Dice Score — without augmentation (CPU)** | 0.79 | 0.50 | 0.72 |
| **Dice Score — with augmentation (CPU)** | 0.74 | 0.8741 | 0.8482 |
| **Dice Score — with augmentation (GPU T4)** | 0.84 | **0.9286** | — |

> ✅ Achieved with only **5 original patients** (expanded to **12 groups** after slicing).
> 🚀 GPU training on Google Colab (Tesla T4) — **200 epochs** — pushed Dice to **0.9286** (state of the art).
> 🧹 Post-processing (largest connected component) maintained high quality at **Dice = 0.8482** on CPU run.
> 📈 Data augmentation improved raw test Dice from **0.50 → 0.9286** (+0.43).

**Segmentation output — after post-processing (with augmentation):**
![Segmentation Result](results/plots/postprocessing_result.png)

**Training curves:**
![Training Curves](results/plots/training_curves.png)

---

## 🗂️ Dataset

| Property | Value |
|----------|-------|
| **Source** | [Medical Segmentation Decathlon](https://medicaldecathlon.com/) — Task 03: Liver |
| **Original patients** | 5 CT scans with liver annotations |
| **After preprocessing** | 12 groups of 75 slices each |
| **Train / Test split** | 10 / 2 groups |
| **Format** | NIfTI (.nii) |

> ⚠️ Dataset **not included** in this repo due to file size. Download it from the official source above.

---

## 🏗️ Pipeline Architecture

```text
5 original patients (DICOM)
        ↓
Preparation_nii.ipynb
  → Split each patient into groups of 75 slices
  → Convert DICOM groups → NIfTI (.nii)
  → Remove empty groups (no liver annotation)
        ↓
PreProcess_train.ipynb
  → Resize: 512×512×75 → 128×128×80 (Spacingd + Resized)
  → Normalize HU intensities: [-200, +200] → [0.0, 1.0]
  → Binarize labels: {0, 1}
  → MONAI transforms: LoadImaged, Spacingd, Orientationd, ScaleIntensityRanged, CropForegroundd, Resized
  → Data Augmentation: RandFlipd (x3), RandRotate90d, RandGaussianNoised
        ↓
Train.ipynb (CPU)
  → U-Net (MONAI) — 3D segmentation
  → Loss: DiceLoss (to_onehot_y=True, sigmoid=True)
  → Optimizer: Adam (lr=1e-4)
  → LR Scheduler: ReduceLROnPlateau (patience=15)
  → 100 epochs on CPU
  → Best model: epoch 96, Dice = 0.8741
        ↓
07-Colab_Training.ipynb (GPU T4) ← NEW
  → Same pipeline adapted for Google Colab
  → CacheDataset (cache_rate=1.0) + batch_size=2 + num_workers=2
  → 200 epochs on Tesla T4
  → Best model: epoch 195, Dice = 0.9286 🏆
        ↓
Testing.ipynb
  → Load best model
  → Evaluate on test set
  → Visualize predictions vs ground truth (32 slices)
        ↓
PostProcessing.ipynb
  → Load raw prediction
  → scipy.ndimage.label → detect all connected components
  → Keep only the largest component (liver)
  → Final Dice Score: 0.8482
```

---

## 🧠 Model Architecture — U-Net

```text
Input (1, 128, 128, 80)
    ↓ Encoder
      Conv3D → ReLU → MaxPool (×4 levels)
    ↓ Bottleneck
      256 feature maps
    ↓ Decoder
      UpConv + Skip Connections (×4 levels)
    ↓ Output
      Conv1×1 → 2 channels (background / liver)
```

**Key concepts:**
- 🔗 **Skip connections** — preserve spatial details lost during downsampling
- 🎯 **Dice Loss** — handles class imbalance (88% background / 12% liver)
- 🔁 **Residual units** — better gradient propagation through the network
- ☁️ **CacheDataset** — preloads full dataset into RAM for faster GPU training

---

## 📈 Data Augmentation

Applied **only on training data** — never on test data.

| Transform | Role | Parameters |
|-----------|------|------------|
| `RandFlipd` (axis=0) | Left/right flip | prob=0.5 |
| `RandFlipd` (axis=1) | Front/back flip | prob=0.5 |
| `RandFlipd` (axis=2) | Up/down flip | prob=0.5 |
| `RandRotate90d` | Random 90°/180°/270° rotation | prob=0.5, max_k=3 |
| `RandGaussianNoised` | Gaussian noise on image only | prob=0.3, std=0.1 |

> Each epoch applies different random transformations → dataset virtually multiplied ×5

---

## ⚙️ Technical Choices

| Parameter | CPU Run | GPU Run (Colab) | Reason |
|-----------|:-------:|:---------------:|--------|
| **Device** | Intel Iris Plus | Tesla T4 | — |
| **Input size** | `128×128×80` | `128×128×80` | RAM constraint |
| **Batch size** | `1` | `2` | GPU has more VRAM |
| **num_workers** | `0` | `2` | Linux allows parallelism |
| **Dataset** | `Dataset` | `CacheDataset` | GPU benefits from preloading |
| **Learning rate** | `1e-4` | `1e-4` | Standard for medical segmentation |
| **LR Scheduler** | `ReduceLROnPlateau` | `ReduceLROnPlateau` | patience=15 |
| **Epochs** | `100` | `200` | GPU allows more epochs |
| **Loss function** | `Dice Loss` | `Dice Loss` | Handles class imbalance |
| **Post-processing** | `scipy.ndimage.label` | — | Removes false positives |

---

## 📁 Repository Structure

```text
Liver_Segmentation/
├── notebooks/
│   ├── Preparation_nii.ipynb       # DICOM → NIfTI + grouping + cleaning
│   ├── PreProcess_train.ipynb      # Resize + normalize + augmentation pipeline
│   ├── Train.ipynb                 # U-Net training loop (CPU)
│   ├── Testing.ipynb               # Evaluation + visualization
│   ├── Utilities.ipynb             # Helper functions (dice_metric, train, show_patient)
│   ├── PostProcessing.ipynb        # Connected components → Dice 0.8482
│   └── 07-Colab_Training.ipynb    # GPU training on Google Colab → Dice 0.9286 🏆
├── sample_data/
│   └── dicom_groups/
│       └── liver_0_0/              # Example: 1 group of 75 DICOM slices
├── results/
│   ├── loss_train.npy
│   ├── loss_test.npy
│   ├── metric_train.npy
│   ├── metric_test.npy
│   └── plots/
│       ├── training_curves.png
│       ├── segmentation_result.png
│       └── postprocessing_result.png
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Local (CPU)

```bash
git clone https://github.com/rowan26/Liver_Segmentation.git
cd Liver_Segmentation
conda create -n dicom_env python=3.8
conda activate dicom_env
pip install -r requirements.txt
```

Run notebooks in order:
```text
Preparation_nii → PreProcess_train → Train → Testing → PostProcessing
```

### Google Colab (GPU) — Recommended

1. Upload your preprocessed `.nii` files to Google Drive:
```text
MyDrive/Liver_Segmentation/preprocessed/Train/images/
MyDrive/Liver_Segmentation/preprocessed/Train/labels/
MyDrive/Liver_Segmentation/preprocessed/Test/images/
MyDrive/Liver_Segmentation/preprocessed/Test/labels/
```
2. Open `notebooks/07-Colab_Training.ipynb` in Google Colab
3. Set Runtime → GPU (T4)
4. Run all cells

---

## 📦 Dependencies

```text
torch
monai
nibabel
numpy
scipy
matplotlib
dicom2nifti
tqdm
```

> Full list available in `requirements.txt`

---

## ⚠️ Known Limitations

- Only **5 original patients** → limited generalization
- Local CPU training → long training time (~1-2h per epoch)
- Label interpolation artifacts during resize (bilinear on binary labels)

---

## 🔮 Future Improvements

- [x] Data augmentation ✅ Dice 0.50 → 0.87
- [x] Post-processing — keep largest connected component ✅ Dice = 0.8482
- [x] Learning Rate Scheduler (ReduceLROnPlateau) ✅ patience=15
- [x] GPU training on Google Colab (Tesla T4) ✅ Dice = 0.9286 🏆
- [ ] More patients (full 130-patient Decathlon dataset)
- [ ] MLflow experiment tracking
- [ ] FastAPI deployment + Docker containerization
- [ ] AWS S3 model storage

---

## 👤 Author

**Rowan Hadjaz**
Cybersecurity & AI Consultant @ Wavestone
AI Engineering Graduate — ISEN JUNIA

[![GitHub](https://img.shields.io/badge/GitHub-rowan26-black?logo=github&style=flat-square)](https://github.com/rowan26)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Rowan%20Hadjaz-blue?logo=linkedin&style=flat-square)](https://www.linkedin.com/in/rowan-hadjaz)
