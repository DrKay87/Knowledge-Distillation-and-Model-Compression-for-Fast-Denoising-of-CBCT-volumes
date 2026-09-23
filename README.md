# Knowledge Distillation and Model Compression for Fast Deep Learning-based Denoising of Cone-beam Computed Tomography Volumes

Official research-code repository accompanying the manuscript by **Khuram Naveed and Ruben Pauwels**, Department of Dentistry and Oral Health, Aarhus University.

This work investigates computationally efficient deep-learning approaches for denoising low-dose dental cone-beam computed tomography (CBCT). A high-capacity **HARU-Net** is used as a teacher to train a lightweight **Residual U-Net (ResU-Net)** student through knowledge distillation. The distilled model is subsequently compressed using structured pruning and half-precision quantization. A separate quantization-aware training (QAT) strategy is also investigated.

> **Status:** The associated manuscript is currently a preprint and has not been peer reviewed.

## Overview

The repository implements three experimental pathways:

```text
                         HARU-Net (teacher)
                                │
Low-dose CBCT ───────────────►  │
                                ▼
High-dose CBCT ─────► KD training of ResU-Net
                                │
                                ▼
                           KD-ResU-Net
                          /           \
                         /             \
       Structured pruning               Quantization-aware
       + KD fine-tuning                  training with KD
               │                              │
       FP16 quantization                      ▼
               │                       qatKD-ResU-Net
               ▼
        qpKD-ResU-Net
```

### 1. Knowledge distillation — KD-ResU-Net

The lightweight ResU-Net student is trained using the pretrained HARU-Net as teacher. The distillation objective combines supervised reconstruction from the high-dose CBCT target with imitation of the teacher prediction:

```text
L = (1 - alpha) * L_hard + alpha * L_soft

L_hard = MSE(student prediction, high-dose target)
L_soft = MSE(student prediction, teacher prediction)
```

During training, `alpha` is progressively increased so that the student initially emphasizes direct learning from the reference data and increasingly incorporates teacher supervision.

### 2. Structured pruning + half-precision quantization — qpKD-ResU-Net

Starting from KD-ResU-Net, structured L2-norm pruning is applied iteratively. Each pruning iteration removes 5% of the least-important channels/weights according to the experimental procedure, followed by 10 epochs of knowledge-distillation fine-tuning at a learning rate of `1e-6`. Five pruning cycles are used in the manuscript (25% cumulative pruning target).

The pruned network is subsequently evaluated using half-precision (FP16) weights/activations, producing **qpKD-ResU-Net**.

### 3. Quantization-aware training — qatKD-ResU-Net

A separate compression route investigates 8-bit quantization-aware training. Fake quantization is introduced during training and the student continues to learn under the knowledge-distillation objective. In the manuscript, QAT is performed for an additional 30 epochs; the first and final convolutional layers are excluded from 8-bit fake quantization.

## Results reported in the manuscript

### Denoising performance

| Model | PSNR (dB) ↑ | SSIM ↑ | GMSD ↓ |
|---|---:|---:|---:|
| Uformer | 36.25 | 0.9447 | 0.1147 |
| SwinIR | 36.12 | 0.9551 | 0.1154 |
| HAT | 36.70 | 0.9569 | 0.1119 |
| HARU-Net | 37.52 | 0.9557 | 0.1084 |
| ResU-Net | 35.03 | 0.9542 | 0.1240 |
| **KD-ResU-Net** | **36.29** | **0.9547** | **0.1121** |
| **qpKD-ResU-Net** | **36.05** | **0.9578** | **0.1144** |
| qatKD-ResU-Net | 33.32 | 0.8099 | 0.2092 |

### Computational efficiency

Measurements reported for an NVIDIA RTX 2080 Ti:

| Model | Parameters | GMACs | 256×256 latency (ms) | 512×512×512 volume (s) |
|---|---:|---:|---:|---:|
| Uformer | 82.251 M | 78.027 | 125.049 | 252.498 |
| SwinIR | 3.133 M | 111.069 | 213.433 | 517.171 |
| HAT | 4.991 M | 174.679 | 670.923 | 777.314 |
| HARU-Net | 92.938 M | 40.760 | 38.830 | 114.918 |
| ResU-Net | 9.502 M | 6.898 | 3.960 | 11.895 |
| **KD-ResU-Net** | **9.502 M** | **6.898** | **8.379** | **19.406** |
| **qpKD-ResU-Net** | **9.502 M** | **6.898** | **3.919** | **10.699** |

The results show that knowledge distillation substantially improves the denoising performance of the lightweight ResU-Net without increasing its parameter count. The pruning and FP16 quantization pipeline further reduces inference time while retaining most of the improvement obtained through distillation. The QAT experiment provides an aggressive low-precision alternative but showed a marked reduction in denoising performance in the reported experiments.

## Repository structure

```text
.
├── README.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── environment.yml
├── .gitignore
├── .gitattributes
│
├── compression/
│   ├── __init__.py
│   ├── data.py
│   ├── losses.py
│   ├── metrics.py
│   ├── pruning.py
│   └── quantization.py
│
├── scripts/
│   └── distill.py
│
├── models/
│   └── README.md
│
├── notebooks/
│   ├── knowledge_distillation.ipynb
│   ├── pruning_after_distillation.ipynb
│   └── quantization_aware_distillation.ipynb
│
├── data/
├── checkpoints/
├── results/
└── assets/
```

## Notebooks

The notebooks preserve the original research workflow and intermediate experimental steps.

- **`knowledge_distillation.ipynb`** — HARU-Net → ResU-Net knowledge distillation and evaluation.
- **`pruning_after_distillation.ipynb`** — iterative structured pruning and KD-based fine-tuning.
- **`quantization_aware_distillation.ipynb`** — QAT combined with teacher-student training.

Local paths in the original research notebooks may need to be updated before execution.

## Model definitions

The experimental notebooks depend on the exact model implementations:

```python
from HARUnet_model_v2_1 import HARU_net
from ResUNet_model import ResUNet
```

These model-definition files were not included in the files used to assemble this release package. Add the exact implementations used in the experiments before claiming full end-to-end reproducibility. The HARU-Net teacher should correspond to the architecture used in the associated HARU-Net work.

## Dataset

The experiments use CBCT scans of **21 human hemimandibular specimens**, acquired using a **3D Accuitomo 170** system at standard adult high-resolution settings (90 kV, 5 mA, 30.8 s). Images were acquired with a 14 × 5 cm field of view and reconstructed to a focused 5 × 5 cm field of view at 0.08 mm isotropic voxel resolution.

The original CBCT dataset is **not distributed with this repository**. Do not commit restricted CBCT data to Git.

## Installation

```bash
git clone https://github.com/DrKay87/Knowledge-Distillation-and-Model-Compression-for-Fast-Denoising-of-CBCT-volumes.git
cd Knowledge-Distillation-and-Model-Compression-for-Fast-Denoising-of-CBCT-volumes

python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

## Example: knowledge-distillation training

After adding the exact HARU-Net and ResU-Net model definitions:

```bash
python scripts/distill.py \
  --train-inputs /path/to/noisy_train.pkl \
  --train-targets /path/to/reference_train.pkl \
  --teacher-checkpoint /path/to/HARUNet_checkpoint.pth \
  --epochs 50 \
  --output checkpoints/KD_ResUNet.pth
```

For exact reproduction of the manuscript experiments, use the corresponding research notebooks and experimental settings.

## Ethics and data privacy

The study used cadaveric human specimens obtained through institutional sources. Specimen handling and imaging were conducted under the Faculty of Dentistry, Chulalongkorn University Human Research Ethics Committee approval/exemption **HREC-DCU 2015-032**. No living human participants were involved, and the data were anonymized.

The `.gitignore` excludes common medical-image formats, datasets, checkpoints, and generated outputs to reduce the risk of accidentally committing restricted data.

## Funding

This research was funded by the **Independent Research Fund Denmark**, project *Synthetic Dental Radiography using Generative Artificial Intelligence*, grant ID **10.46540/3165-00237B**.

## Citation

The manuscript is currently available as a preprint. Please cite the preprint when using this code and update the citation if a peer-reviewed version becomes available.

**Naveed, K., & Pauwels, R.**  
*Knowledge Distillation and Model Compression for Fast Deep Learning-based Denoising of Cone-beam Computed Tomography Volumes.*  
Preprint, 2026.

## Related work

The teacher model is based on:

**K. Naveed and R. Pauwels**, *HARU-Net: Hybrid Attention Residual U-Net for Edge-Preserving Denoising in Cone-Beam Computed Tomography.*

## License

The repository code is provided under the MIT License. Dataset access, third-party implementations, pretrained weights, and other external resources may be subject to separate licenses or usage restrictions.
