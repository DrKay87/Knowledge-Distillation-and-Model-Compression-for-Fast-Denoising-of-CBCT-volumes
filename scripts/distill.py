"""Train the ResU-Net student from a pretrained HARU-Net teacher.

The default alpha schedule follows the manuscript: alpha increases linearly
through training so early epochs emphasize the reference target and later
epochs increasingly emphasize teacher supervision.

Requires the exact model definitions used in the experiments:
    models/HarUnet_model.py
    models/ResUnet_model.py
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from compression.data import load_data, CBCTDataset
from compression.losses import DistillationLoss

from models.HarUnet_model import HARU_net
from models.ResUnet_model import ResUNet
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-inputs", required=True)
    p.add_argument("--train-targets", required=True)
    p.add_argument("--teacher-checkpoint", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Fixed alpha. If omitted, use a linear 0-to-1 schedule.",
    )
    p.add_argument("--output", default="checkpoints/KD_ResUNet.pth")
    return p.parse_args()


def main():
    a = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CBCTDataset(load_data(a.train_inputs), load_data(a.train_targets))
    loader = DataLoader(dataset, batch_size=a.batch_size, shuffle=True)

    teacher = HARU_net().to(device)
    state = torch.load(a.teacher_checkpoint, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    teacher.load_state_dict(state)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False

    student = ResUNet().to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=a.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5
    )
    criterion = DistillationLoss()

    for epoch in range(1, a.epochs + 1):
        student.train()
        running = 0.0

        # Manuscript schedule: progressively increase teacher contribution.
        alpha = a.alpha if a.alpha is not None else (epoch - 1) / max(a.epochs - 1, 1)

        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            with torch.no_grad():
                teacher_outputs = teacher(inputs)

            student_outputs = student(inputs)
            loss = criterion(
                student_outputs, teacher_outputs, targets, alpha=alpha
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item() * inputs.size(0)

        epoch_loss = running / len(dataset)
        scheduler.step(epoch_loss)
        print(
            f"Epoch {epoch:03d}/{a.epochs} | "
            f"alpha={alpha:.4f} | loss={epoch_loss:.8f}"
        )

    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(student.state_dict(), a.output)
    print(f"Saved distilled student to {a.output}")


if __name__ == "__main__":
    main()
