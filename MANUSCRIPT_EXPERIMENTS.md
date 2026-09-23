# Manuscript experiment settings

This file summarizes implementation-critical settings stated in the accompanying preprint.

## Knowledge distillation

- Teacher: HARU-Net
- Student: ResU-Net
- Resulting model: KD-ResU-Net
- Hard loss: MSE between student output and high-dose/reference target
- Soft loss: MSE between student output and teacher output
- Combined loss: `(1-alpha) * hard_loss + alpha * soft_loss`
- Optimizer: Adam
- Initial learning rate: `1e-4`
- Learning-rate reduction: factor `0.1` after validation loss fails to decrease for five consecutive epochs
- Student training: 50 epochs
- Alpha: progressively increased from 0 toward 1 during training

## Structured pruning + quantization

- Starting model: KD-ResU-Net
- Criterion: L2-norm structured pruning
- Pruning increment: 5% per iteration
- Number of pruning cycles: 5
- Cumulative pruning target reported: 25%
- Fine-tuning after each pruning step: 10 epochs
- Fine-tuning learning rate: `1e-6`
- Fine-tuning objective: knowledge-distillation loss
- Final precision stage: FP16
- Resulting model: qpKD-ResU-Net

## Quantization-aware training

- Starting framework: KD teacher-student framework
- Precision: 8-bit fake quantization
- Quantized: convolutional weights and activations
- Exceptions: first and final layers
- Additional QAT training: 30 epochs
- Resulting model: qatKD-ResU-Net

## Benchmark hardware

Inference latency reported in the manuscript was measured using an NVIDIA RTX 2080 Ti.

## Data

- 21 human hemimandibular specimens
- Scanner: 3D Accuitomo 170
- 90 kV, 5 mA, 30.8 s
- Acquisition FOV: 14 x 5 cm
- Focused reconstruction FOV: 5 x 5 cm
- Isotropic voxel resolution: 0.08 mm
