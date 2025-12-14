import os
import argparse
from glob import glob
import random

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as T
import matplotlib.pyplot as plt


# -----------------------------------------------------------
#  Dataset
# -----------------------------------------------------------

class MidAirDepthDataset(Dataset):
    """
    Expects directory structure like:

    root_dir/
        color_left/
            trajectory_0000/frames/000000.JPG
            ...
        depth/
            trajectory_0000/frames/000000.PNG
            ...

    RGB:   .JPG / .JPEG
    Depth: .PNG (8-bit or 16-bit single-channel)
    """

    def __init__(self, root_dir, resize_hw=(192, 320), train=True, max_samples=None):
        super().__init__()
        self.root_dir = root_dir
        self.color_root = os.path.join(root_dir, "color_left")
        self.depth_root = os.path.join(root_dir, "depth")
        self.resize_hw = resize_hw  # (H, W)
        self.train = train

        self.samples = self._build_index()

        if max_samples is not None and len(self.samples) > max_samples:
            self.samples = random.sample(self.samples, max_samples)

        H, W = resize_hw

        # RGB transforms: Aug B if train, lighter if val
        if self.train:
            self.rgb_transform = T.Compose([
                T.Resize((H, W), interpolation=T.InterpolationMode.BILINEAR),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.15
                ),
                T.ToTensor(),
            ])
        else:
            self.rgb_transform = T.Compose([
                T.Resize((H, W), interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
            ])

    def _build_index(self):
        samples = []

        traj_dirs = sorted(glob(os.path.join(self.color_root, "trajectory_*")))
        if not traj_dirs:
            raise RuntimeError(f"No trajectory_* folders found under {self.color_root}")

        for traj in traj_dirs:
            frames_dir = os.path.join(traj, "frames")

            rgb_files = sorted(
                glob(os.path.join(frames_dir, "*.JPG")) +
                glob(os.path.join(frames_dir, "*.JPEG")) +
                glob(os.path.join(frames_dir, "*.jpg")) +
                glob(os.path.join(frames_dir, "*.jpeg"))
            )

            for rgb_path in rgb_files:
                rel = os.path.relpath(rgb_path, self.color_root)  # trajectory_xxxx/frames/000000.JPG
                base_no_ext = os.path.splitext(rel)[0]
                depth_rel = base_no_ext + ".PNG"
                depth_path = os.path.join(self.depth_root, depth_rel)
                if os.path.exists(depth_path):
                    samples.append((rgb_path, depth_path))

        if not samples:
            raise RuntimeError("No RGB/depth pairs found. Check directory structure.")

        print(f"📌 Successfully paired: {len(samples)} images")
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_depth(self, depth_path):
        """
        Robust depth loading:
        - supports 8-bit and 16-bit PNG
        - normalizes to [0,1]
        - resizes to self.resize_hw with nearest-neighbor
        """
        depth_img = Image.open(depth_path)

        depth_np = np.array(depth_img)
        depth_np = depth_np.astype(np.float32)

        # If 16-bit or larger range, normalize by 65535 or max
        if depth_np.max() > 255:
            depth_np /= 65535.0
        else:
            depth_np /= 255.0

        # to tensor [1, H, W]
        depth = torch.from_numpy(depth_np).unsqueeze(0)  # [1, H0, W0]

        # resize to target size
        H, W = self.resize_hw
        depth = F.interpolate(
            depth.unsqueeze(0),  # [1,1,H0,W0]
            size=(H, W),
            mode="nearest"
        ).squeeze(0)  # [1,H,W]

        return depth

    def __getitem__(self, idx):
        rgb_path, depth_path = self.samples[idx]

        rgb_img = Image.open(rgb_path).convert("RGB")
        rgb = self.rgb_transform(rgb_img).float()  # [3,H,W]

        depth = self._load_depth(depth_path).float()  # [1,H,W]

        return rgb, depth


# -----------------------------------------------------------
#  Model (simple U-Net-like)
# -----------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DepthUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base_ch=32):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8)

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.out_conv(d1)
        out = torch.sigmoid(out)  # normalized depth [0,1]
        return out


# -----------------------------------------------------------
#  Loss B: L1 + SSIM
# -----------------------------------------------------------

def ssim(pred, target, window_size=3):
    """
    Simple differentiable SSIM for single-channel images in [0,1].
    pred, target: [B,1,H,W]
    returns mean SSIM over batch.
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(pred, window_size, 1, 0)
    mu_y = F.avg_pool2d(target, window_size, 1, 0)

    sigma_x = F.avg_pool2d(pred * pred, window_size, 1, 0) - mu_x ** 2
    sigma_y = F.avg_pool2d(target * target, window_size, 1, 0) - mu_y ** 2
    sigma_xy = F.avg_pool2d(pred * target, window_size, 1, 0) - mu_x * mu_y

    ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    ssim_map = ssim_n / (ssim_d + 1e-7)
    return ssim_map.mean()


def depth_loss(pred, target):
    """
    Loss B: 0.85 * L1 + 0.15 * (1 - SSIM)
    """
    l1 = torch.mean(torch.abs(pred - target))
    ssim_val = ssim(pred, target)
    loss = 0.85 * l1 + 0.15 * (1.0 - ssim_val)
    return loss


# -----------------------------------------------------------
#  Training / validation loops
# -----------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    for rgb, depth in loader:
        rgb = rgb.to(device)
        depth = depth.to(device)

        optimizer.zero_grad()
        pred = model(rgb)
        loss = depth_loss(pred, depth)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * rgb.size(0)

    return running_loss / len(loader.dataset)


def validate_one_epoch(model, loader, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for rgb, depth in loader:
            rgb = rgb.to(device)
            depth = depth.to(device)

            pred = model(rgb)
            loss = depth_loss(pred, depth)
            running_loss += loss.item() * rgb.size(0)

    return running_loss / len(loader.dataset)


# -----------------------------------------------------------
#  Visualization helpers
# -----------------------------------------------------------

def save_loss_plot(train_losses, val_losses, out_path):
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_losses, label="train")
    plt.plot(epochs, val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Depth Loss (L1 + SSIM)")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_depth_examples(model, dataset, device, out_path, num_examples=10):
    model.eval()
    indices = np.random.choice(len(dataset), size=min(num_examples, len(dataset)), replace=False)

    n = len(indices)
    plt.figure(figsize=(6, 2 * n))

    with torch.no_grad():
        for row, idx in enumerate(indices):
            rgb, depth = dataset[idx]
            rgb_in = rgb.unsqueeze(0).to(device)
            pred = model(rgb_in).cpu().squeeze(0)  # [1,H,W]

            depth_np = depth.squeeze(0).numpy()
            pred_np = pred.squeeze(0).numpy()

            # use GT min/max for both GT & pred so color scale is comparable
            vmin = float(depth_np.min())
            vmax = float(depth_np.max())

            col_base = 3  # RGB, GT, Pred

            # RGB
            ax = plt.subplot(n, col_base, row * col_base + 1)
            ax.imshow(np.transpose(rgb.numpy(), (1, 2, 0)))
            ax.set_title(f"RGB (idx {idx})", fontsize=8)
            ax.axis("off")

            # GT depth
            ax = plt.subplot(n, col_base, row * col_base + 2)
            im = ax.imshow(depth_np, cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title("GT depth", fontsize=8)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.01)

            # Pred depth
            ax = plt.subplot(n, col_base, row * col_base + 3)
            im = ax.imshow(pred_np, cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title("Pred depth", fontsize=8)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.01)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------------------------------------
#  Main
# -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Path to MidAir subset root (e.g. .../Kite_training/sunny)")
    parser.add_argument("--log_dir", type=str, default="runs/final_part1")
    parser.add_argument("--outputs_dir", type=str, default="outputs_final_part1")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_train_samples", type=int, default=800,
                        help="Limit number of training samples (for speed)")
    parser.add_argument("--max_val_samples", type=int, default=200,
                        help="Limit number of validation samples (for speed)")
    args = parser.parse_args()

    os.makedirs(args.outputs_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----- dataset -----
    full_dataset = MidAirDepthDataset(
        root_dir=args.root_dir,
        resize_hw=(192, 320),
        train=True,
        max_samples=None
    )

    total_len = len(full_dataset)
    val_len = int(0.2 * total_len)
    train_len = total_len - val_len
    train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])

    # Aug B for train, light aug for val
    train_dataset.dataset.train = True
    train_dataset.dataset.rgb_transform = MidAirDepthDataset(
        args.root_dir, resize_hw=(192, 320), train=True
    ).rgb_transform

    val_dataset.dataset.train = False
    val_dataset.dataset.rgb_transform = MidAirDepthDataset(
        args.root_dir, resize_hw=(192, 320), train=False
    ).rgb_transform

    # optionally limit samples for speed
    if args.max_train_samples is not None and train_len > args.max_train_samples:
        train_dataset.indices = train_dataset.indices[:args.max_train_samples]
        train_len = len(train_dataset)

    if args.max_val_samples is not None and val_len > args.max_val_samples:
        val_dataset.indices = val_dataset.indices[:args.max_val_samples]
        val_len = len(val_dataset)

    print(f"Total samples: {total_len}")
    print(f"Train samples: {train_len}, Val samples: {val_len}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ----- model, optimizer -----
    model = DepthUNet(in_ch=3, out_ch=1, base_ch=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_val = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:03d} | train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f}")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(args.outputs_dir, "best_model.pt")
            torch.save(model.state_dict(), best_path)
            print(f"  ✅ New best val loss. Model saved to {best_path}")

    writer.close()

    # ----- final visualizations -----
    loss_plot_path = os.path.join(args.outputs_dir, "loss_curve.png")
    save_loss_plot(train_losses, val_losses, loss_plot_path)
    print(f"Saved loss curve to {loss_plot_path}")

    examples_path = os.path.join(args.outputs_dir, "depth_predictions_grid.png")
    save_depth_examples(model, val_dataset, device, examples_path, num_examples=10)
    print(f"Saved depth prediction examples to {examples_path}")


if __name__ == "__main__":
    main()