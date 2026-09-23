"""Losses used in the HARU-Net -> ResUNet knowledge-distillation experiments."""
import torch.nn as nn

class DistillationLoss(nn.Module):
    """Weighted teacher-matching and ground-truth reconstruction loss.

    loss = alpha * MSE(student, teacher) + (1-alpha) * MSE(student, target)
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, student_out, teacher_out, targets, alpha=0.5):
        soft_loss = self.mse(student_out, teacher_out.detach())
        hard_loss = self.mse(student_out, targets)
        return alpha * soft_loss + (1.0 - alpha) * hard_loss
