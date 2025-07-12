# 基于传统机器学习与深度学习的遥感图像分类

本项目旨在探索传统特征工程方法与现代深度学习模型在遥感图像分类任务中的性能差异，使用 [SkyView Aerial Landscape Dataset](https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset)，数据集包含15类均衡场景，总计12,000张高清航拍图像。

---

## ✨ 项目亮点

- 📷 数据集：15类遥感图像，每类800张，分布均衡
- 🧠 模型方法：
  - **传统方法**：SIFT + LBP 特征 + SVM / KNN 分类器
  - **深度学习**：ResNet-18 与 EfficientNet-B0（迁移学习）
- 🔍 可解释性分析：集成 **Grad-CAM** 可视化模型关注区域
- 🤖 GPT-4o：用于模型误判样本的语义级别再评估
- 📈 最佳准确率：**98.5%**（EfficientNet-B0）

---

## 📁 项目结构

```
.
├── Data.ipynb                      # 数据分析与增强流程
├── Notebook_SVM.ipynb             # SIFT+LBP+SVM/KNN 分类器实现
├── Notebook_ResNet18.ipynb        # ResNet-18 迁移学习模型训练
├── Notebook_EfficientNet-B0.ipynb # EfficientNet-B0 模型训练与评估
├── Notebook_GradCAM.ipynb         # Grad-CAM 可视化
├── report.pdf                      # 项目最终报告（IEEE格式）
└── README.md                       # 本文件
```

---

## 🔧 环境配置

安装依赖：

```bash
pip install torch torchvision opencv-python pandas scikit-learn matplotlib seaborn tqdm
```

或通过 `requirements.txt` 文件一键安装：

```bash
pip install -r requirements.txt
```

### 验证环境

- Python 3.10+
- Ubuntu 22.04 + CUDA 12.1
- GPU: RTX 3090 (24GB)

---

## 📊 实验结果

| 模型类型          | 准确率   | 精确率   | 召回率   | F1 分数  |
|-------------------|----------|----------|----------|----------|
| SVM (SIFT+LBP)    | 71.8%    | 0.7184   | 0.7177   | 0.7167   |
| ResNet-18         | 91.7%    | 0.9188   | 0.9171   | 0.9171   |
| EfficientNet-B0   | **98.5%**| **0.9870**| 0.9851   | 0.9850   |

- ResNet-18 在中等规模数据集上具有良好泛化能力
- EfficientNet-B0 在精度与效率之间取得最优平衡

---

## 🧪 模型评估方式

- **准确率、精确率、召回率、F1分数**（macro平均）
- **混淆矩阵**：评估模型分类混淆情况
- **训练曲线**：训练 / 验证准确率与损失
- **Grad-CAM**：可视化模型注意区域
- **GPT-4o 语义审查**：错误样本的语义解释辅助判断

---

## 📌 模型对比分析

- **SVM**：轻量、可解释，但难以处理复杂视觉类别
- **ResNet-18**：表现稳定，适用于大多数视觉任务
- **EfficientNet-B0**：结构高效，精度最优，适合部署
- **GPT-4o**：在语义层面提供辅助解释，准确率较低（9.3%）

---

## 🧠 Grad-CAM 可视化示例

| 场景类别 | EfficientNet-B0 关注区域 | ResNet-18 关注区域         |
|----------|--------------------------|------------------------------|
| Railway  | 铁轨与站台               | 周边地形                    |
| Port     | 船坞与码头               | 背景水域                    |
| Airport  | 航站楼与跑道核心区域     | 宽泛基础设施                |
| Highway  | 中心道路结构             | 附近建筑与车道              |
| Parking  | 密集车辆区域             | 周边空旷路面                |

---

## 🔍 运行说明

1. 下载 [SkyView 数据集](https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset)
2. 将图像整理为：
   - ResNet 使用 `ImageFolder` 结构（train/test 分文件夹）
   - EfficientNet 使用带有路径的 CSV 文件（augmented_train.csv 等）
3. 按顺序运行对应 Jupyter Notebook 文件
4. 使用 `Notebook_GradCAM.ipynb` 可视化注意区域

---

## 🔮 未来工作方向

- 使用类权重 / Focal Loss 等方法应对类别不平衡
- 探索轻量化架构：MobileNetV3、ShuffleNetV2
- 模型压缩：Knowledge Distillation 实现高效部署
- 融合大模型：CLIP、GPT-4、Flamingo 实现语义增强
- 加入 SHAP/LIME 等解释性方法提升可理解性

---

## 📚 项目引用

- [SkyView 数据集 (Kaggle)](https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset)
- [EfficientNet (ICML 2019)](https://arxiv.org/abs/1905.11946)
- [Grad-CAM (ICCV 2017)](https://arxiv.org/abs/1610.02391)
- [PyTorch 官网](https://pytorch.org/)

---

## 🙏 致谢

感谢以下开源工具的支持：

- **PyTorch / TorchVision**
- **OpenCV / Scikit-learn**
- **HuggingFace / GPT-4**
- **Matplotlib / Seaborn / tqdm**

为本项目的实现提供了坚实的基础。



# Aerial Scene Classification using Traditional ML and Deep Learning

This project investigates aerial scene classification using both handcrafted features with traditional classifiers and modern deep learning approaches. We experiment on the [SkyView Aerial Landscape Dataset](https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset), which consists of 12,000 images across 15 balanced scene categories.

---

## ✨ Highlights

- 📷 Dataset: 15-class aerial image dataset (800 images/class, balanced)
- 🧠 Models:
  - **Traditional ML**: SIFT + LBP + SVM/KNN
  - **Deep Learning**: ResNet-18 and EfficientNet-B0 (Transfer Learning)
- 🔍 Interpretability: Integrated **Grad-CAM** for visualizing model focus
- 🤖 GPT-4o: Used for post-hoc semantic re-evaluation of misclassifications
- 📈 Best Accuracy: **98.5%** using EfficientNet-B0

---

## 📁 Folder Structure

```
.
├── Data.ipynb                      # EDA, visualization & augmentation pipeline
├── Notebook_SVM.ipynb             # SIFT + LBP + SVM/KNN classifiers
├── Notebook_ResNet18.ipynb        # Transfer learning with ResNet-18
├── Notebook_EfficientNet-B0.ipynb # EfficientNet-B0 training via CSV
├── Notebook_GradCAM.ipynb         # Grad-CAM visualization for both DL models
├── report.pdf                     # Final IEEE-style project report
└── README.md                      # This file
```

---

## 🔧 Setup

Install all dependencies:

```bash
pip install torch torchvision opencv-python pandas scikit-learn matplotlib seaborn tqdm
```

Or install via `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Verified on

- Python 3.10+
- Ubuntu 22.04 + CUDA 12.1
- GPU: RTX 3090 (24GB)

---

## 📊 Results

| Model            | Accuracy | Precision | Recall | F1 Score |
|------------------|----------|-----------|--------|----------|
| SVM (SIFT+LBP)   | 71.8%    | 0.7184    | 0.7177 | 0.7167   |
| ResNet-18        | 91.7%    | 0.9188    | 0.9171 | 0.9171   |
| EfficientNet-B0  | **98.5%**| **0.9870**| 0.9851 | 0.9850   |

- ResNet-18 generalizes well with 25 epochs of fine-tuning.
- EfficientNet-B0 achieves state-of-the-art performance even with fewer parameters.

---

## 🧪 Evaluation Methods

- **Precision / Recall / F1-score** (macro-averaged)
- **Confusion Matrices** for per-class analysis
- **Training curves** (Accuracy & Loss)
- **Grad-CAM** heatmaps for visual explanations
- **GPT-4o Review** on misclassified images for semantic inspection

---

## 📌 Model Insights

- **SVM**: Fast, interpretable, fails on visually similar scenes (e.g. Lake vs River)
- **ResNet-18**: Good generalization, struggles on fine-grained region focus
- **EfficientNet-B0**: Best accuracy and focus (verified via Grad-CAM)
- **GPT-4o**: Provided semantic reasoning but had low visual accuracy (~9.3%), showing potential for hybrid AI

---

## 🧠 Grad-CAM Visualizations

Class-specific focus differences:

| Scene     | EfficientNet Focus     | ResNet Focus               |
|-----------|------------------------|----------------------------|
| Railway   | Tracks and platforms   | Broad terrain              |
| Port      | Ships & docks          | Water background           |
| Highway   | Lane centerlines       | Roadside + buildings       |
| Airport   | Terminal core          | Surrounding infrastructure |
| Parking   | Vehicle clusters       | Empty road spaces          |

---

## 🔍 How to Run

1. Download the [SkyView Dataset](https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset)
2. Prepare images in `ImageFolder` format for ResNet; CSV format for EfficientNet
3. Run respective notebooks step by step
4. Run `Notebook_GradCAM.ipynb` to visualize model attention maps
5. Run `Data.ipynb` to explore the dataset and augmentation pipeline

---

## 🧬 Future Work

- Class-weighted loss, focal loss for long-tail class scenarios
- Lightweight model exploration (MobileNetV3, ShuffleNetV2)
- Model compression via knowledge distillation
- Hybrid pipelines: CNN + LLM (e.g., CLIP, GPT-4, Flamingo)
- Explainable AI: Integrate SHAP, LIME into visual diagnosis

---

## 📚 References

- [SkyView Dataset on Kaggle](https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset)
- [EfficientNet (ICML 2019)](https://arxiv.org/abs/1905.11946)
- [Grad-CAM (ICCV 2017)](https://arxiv.org/abs/1610.02391)
- [PyTorch & TorchVision Docs](https://pytorch.org/vision/stable/index.html)

---

## 🙏 Acknowledgements

Thanks to PyTorch, OpenCV, scikit-learn, HuggingFace, and OpenAI GPT-4 for open-access tools and models that enabled this research.
