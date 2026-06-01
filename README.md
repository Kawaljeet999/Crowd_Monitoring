# Privacy-Preserving Crowd Analysis Using SEMT

## Overview

This repository contains the implementation of a privacy-preserving crowd analysis framework based on the **Stencil Elliptical Masking Technique (SEMT)**. The system anonymizes identity-sensitive regions in surveillance images while preserving crowd structure for accurate crowd counting and density estimation.

The framework combines SEMT-based anonymization with a ResNet-50 and Faster R-CNN architecture to achieve a balance between privacy protection and analytical utility.

---

## Features

* Privacy-preserving crowd analysis
* Stencil Elliptical Masking Technique (SEMT)
* Adaptive elliptical anonymization
* Crowd counting and density estimation
* ResNet-50 backbone
* Faster R-CNN detector
* Support for multiple benchmark datasets

---

## Datasets

The framework supports:

* ShanghaiTech Part A
* ShanghaiTech Part B
* Mall Dataset
* UCSD Dataset

---

## Project Structure

```text
Privacy-Preserving-Crowd-Analysis/
│
├── datasets/
│   ├── ShanghaiTech/
│   ├── Mall/
│   └── UCSD/
│
├── semt/
│   ├── masking.py
│   └── anonymization.py
│
├── models/
│   ├── resnet50.py
│   └── faster_rcnn.py
│
├── training/
│   ├── train.py
│   └── evaluate.py
│
├── results/
│
├── requirements.txt
│
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/username/privacy-preserving-crowd-analysis.git

cd privacy-preserving-crowd-analysis
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

Linux/macOS:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Requirements

```txt
torch
torchvision
numpy
opencv-python
matplotlib
scikit-learn
pillow
tqdm
```

---

## Training

Train the model using:

```bash
python training/train.py
```

---

## Evaluation

Evaluate the trained model:

```bash
python training/evaluate.py
```

---

## Methodology

1. Load crowd images and annotations.
2. Detect human head locations.
3. Generate adaptive elliptical masks.
4. Apply SEMT anonymization.
5. Create privacy-compliant training data.
6. Train ResNet-50 + Faster R-CNN.
7. Generate crowd counts and density maps.
8. Evaluate performance using privacy and utility metrics.

---

## Evaluation Metrics

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Average Precision (AP)
* Pixel Anonymization Ratio (PAR)

---

## Results

The proposed SEMT framework achieves:

| Method                  | AP (%) |
| ----------------------- | ------ |
| Blurring                | 87.51  |
| Pixelization            | 91.17  |
| DeepPrivacy             | 93.61  |
| Adversarial Obfuscation | 89.26  |
| SEMT (Proposed)         | 95.50  |

The results demonstrate that SEMT preserves privacy while maintaining high crowd-counting performance.

---

## Authors

Kawaljeet Singh
Laxmi Choudhary
Vineet Sharma

---

## Citation

If you use this work in your research, please cite the corresponding paper.

---

## License

This project is released for academic and research purposes.
