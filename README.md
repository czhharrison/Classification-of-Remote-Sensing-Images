# COMP9517 Group Project: Aerial Scene Classification

This project explores aerial scene classification using both traditional machine learning and deep learning techniques. The dataset used is the [SkyView Aerial Landscape Dataset](https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset), which contains 15 balanced scene categories with 800 images each.

---

## Folder Structure

```
.
├── Notebook_SVM.ipynb            # Traditional method using SIFT + LBP + SVM
├── Notebook_ResNet18.ipynb       # CNN-based classification using ResNet-18
├── Notebook_EfficientNet-B0.ipynb# EfficientNet-B0 with custom CSV-based dataset loader
├── Notebook_GradCAM.ipynb        # Grad-CAM attention visualisation (ResNet-18 & EfficientNet-B0)
├── Data.ipynb                    # Exploratory data analysis and preprocessing
├── report.pdf                    # Project report (IEEE format)
└── README.md                     # This file
```

---

## Requirements

Install dependencies using pip:

```bash
pip install torch torchvision opencv-python pandas scikit-learn matplotlib seaborn tqdm
```
## Dependencies  (verified on Ubuntu 22.04 + CUDA 12.1)

| Package | Version |
|---------|---------|
| python  | 3.10+   |
| torch   | 2.2.1   |
| torchvision | 0.17 |
| opencv-python | 4.10 |
| scikit-learn | 1.5.0 |
| pandas  | 2.2.2 |
| matplotlib | 3.9.0 |
| seaborn | 0.13.2 |
| tqdm    | 4.66.3 |

> **Tip:** You may also ship a `requirements.txt` file and run  
> `pip install -r requirements.txt` for one-click installation.

---

## Reproducibility

- **Random seed**: `torch.manual_seed(42)`, `numpy.random.seed(42)`  
- **Hardware**: RTX 3090 (24 GB); SVM < 1 min, ResNet-18 ≈ 10 min, EfficientNet-B0 ≈ 10 min  
- **Exact commands**
  ```bash
  # ResNet-18
  python train_resnet18.py --epochs 25 --batch_size 32 --lr 1e-4 --seed 42

  # EfficientNet-B0
  python train_efficientnet.py --epochs 25 --batch_size 32 --lr 1e-4 --seed 42

---

## How to Run

### SVM Notebook

1. Place original dataset under `./Aerial_Landscapes/`, structured as 15 subfolders (Airport, Beach, ...).
2. Open `Notebook_SVM.ipynb` and run all cells sequentially.

### ResNet18 Notebook

1. Dataset must be under `./Aerial_Landscapes/train/` and `./Aerial_Landscapes/test/` in `ImageFolder` format.
2. Open `Notebook_ResNet18.ipynb` and execute all cells.

### EfficientNet-B0 Notebook

1. Ensure CSV files `augmented_train.csv` and `test.csv` are placed under `./COMP9517/`.
2. Image paths in CSV should be relative to `COMP9517/` directory.
3. Open `Notebook_EfficientNet-B0.ipynb` and run all cells.

### Grad-CAM Notebook
1. Put trained checkpoints into ./checkpoints/ (filenames configurable).
2. Open Notebook_GradCAM.ipynb ➜ select model ➜ run all cells.
3. Heat-map PNGs will be written to ./cam_vis/ for easy inspection.

### Data Notebook
1. Open Data.ipynb to view the exploratory data analysis, preprocessing steps, and data augmentation pipeline.
2. This notebook demonstrates the dataset distribution, class balance, and shows sample visualizations before and after augmentation.
---

## Results Overview

| Model            | Accuracy | Precision | Recall | F1 Score |
|------------------|----------|-----------|--------|----------|
| SVM (SIFT+LBP)   | ~71.8%   | ~0.72     | ~0.72  | ~0.72    |
| ResNet-18        | ~91.7%   | ~0.92     | ~0.92  | ~0.92    |
| EfficientNet-B0  | ~98.5%   | ~0.99     | ~0.99  | ~0.99    |

---



###

```
---

## Acknowledgements

- **PyTorch** [Paszke et al., 2019]  
- **TorchVision** (pre-trained ResNet-18 & EfficientNet-B0)  
- **OpenCV** (SIFT / LBP)  
- **scikit-learn** (SVM & KNN)

All above libraries are released under permissive open-source
licenses (BSD / MIT / Apache).

---

## Citation

If you use this repository in academic work, please cite our report:

> C. Cui *et al.*, “Comparative Study of Traditional Machine Learning and  
> Deep Learning Methods for Aerial Scene Classification Using the SkyView  
> Dataset,” COMP9517 Group Project Report, UNSW, 2025.
---
## References

- [SkyView Dataset on Kaggle](https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset)
- PyTorch & torchvision documentation
- OpenCV and scikit-learn official documentation

## CSV Download (EfficientNet-B0)

The required CSV files for EfficientNet-B0 are not included due to file size limits.

You can download them here:  
[Google Drive - CSV Files](https://drive.google.com/drive/folders/18FuRKMdjh8qK3sf65K6K4RizObc7L7oE?usp=sharing)

After downloading:
- Place `augmented_train.csv` and `test.csv` inside a folder named `COMP9517/`
- Make sure the image paths referenced in the CSV point to valid files from the SkyView dataset

