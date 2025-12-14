import os
import argparse
from glob import glob
import random

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as T
import matplotlib.pyplot as plt


# ===========================================================
#  Dataset for UseGeo
# ===========================================================

class UseGeoDepthDataset(Dataset):
    """
    Expects directory structure like:

    root_dir/
      Dataset-1/
        undistorted_images/
        depth_maps/
      Dataset-2/
        undistorted_images/
        depth_maps/
      Dataset-3/
        undistorted_images/
        depth_maps/
    """

    def __init__(self, root_dir, resize_hw=(192, 320), train=True, max_samples=None):
        super().__init__()
        self.root_dir = root_dir
        self.resize_hw = resize_hw
        self.train = train

        self.samples = self._build_index()

        if max_samples is not None and len(self.samples) > max_samples:
            self.samples = random.sample(self.samples, max_samples)

        H, W = resize_hw

        if self.train:
            self.rgb_transform = T.Compose([
                T.Resize((H, W), interpolation=T.InterpolationMode.BILINEAR),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
                T.ToTensor(),
            ])
        else:
            self.rgb_transform = T.Compose([
                T.Resize((H, W), interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
            ])

    def _build_index(self):
        samples = []

        dataset_dirs = sorted(
            d for d in glob(os.path.join(self.root_dir, "Dataset-*"))
            if os.path.isdir(d)
        )

        if not dataset_dirs:
            raise RuntimeError(
                f"No Dataset-* folders found under {self.root_dir}. "
                "Expected e.g. Dataset-1, Dataset-2, Dataset-3."
            )

        rgb_exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff",
                    ".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF")
        depth_exts = rgb_exts

        for ds in dataset_dirs:
            rgb_dir = os.path.join(ds, "undistorted_images")
            depth_dir = os.path.join(ds, "depth_maps")
            if not os.path.isdir(depth_dir):
                alt_depth = os.path.join(ds, "Depth_resized")
                if os.path.isdir(alt_depth):
                    depth_dir = alt_depth

            if not os.path.isdir(rgb_dir) or not os.path.isdir(depth_dir):
                print(f"⚠️ Skipping {ds}: missing undistorted_images/ or depth_maps/")
                continue

            rgb_dict = {}
            for rp in glob(os.path.join(rgb_dir, "*")):
                if not rp.lower().endswith(rgb_exts):
                    continue
                base = os.path.splitext(os.path.basename(rp))[0]
                rgb_dict[base] = rp

            for dp in glob(os.path.join(depth_dir, "*")):
                if not dp.lower().endswith(depth_exts):
                    continue
                depth_base = os.path.splitext(os.path.basename(dp))[0]

                # remove one "depth_" occurrence to match the RGB filename
                rgb_key = depth_base.replace("depth_", "", 1)

                if rgb_key in rgb_dict:
                    samples.append((rgb_dict[rgb_key], dp))

        if not samples:
            raise RuntimeError(
                "No RGB/depth pairs found in UseGeo root. "
                "Check that filenames match between undistorted_images/ and depth_maps/."
            )

        print(f"📌 UseGeo: successfully paired {len(samples)} RGB/depth images")
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_depth(self, depth_path):
        depth_img = Image.open(depth_path)
        depth_np = np.array(depth_img).astype(np.float32)

        if depth_np.size == 0:
            raise RuntimeError(f"Empty depth map at {depth_path}")

        max_val = float(depth_np.max())
        if max_val <= 0:
            max_val = 1.0
        depth_np /= max_val

        depth = torch.from_numpy(depth_np).unsqueeze(0)  # [1,H0,W0]
        H, W = self.resize_hw
        depth = F.interpolate(
            depth.unsqueeze(0),
            size=(H, W),
            mode="nearest"
        ).squeeze(0)  # [1,H,W]
        return depth

    def __getitem__(self, idx):
        rgb_path, depth_path = self.samples[idx]

        rgb_img = Image.open(rgb_path).convert("RGB")
        rgb = self.rgb_transform(rgb_img).float()

        depth = self._load_depth(depth_path).float()
        return rgb, depth


# ===========================================================
#  Model (same as Part 1)
# ===========================================================

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

        self.enc1 = ConvBlock(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_ch * 4, base_ch * 8)

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
        out = torch.sigmoid(out)
        return out


# ===========================================================
#  Loss B: L1 + SSIM
# ===========================================================

def ssim(pred, target, window_size=3):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(pred, window_size, 1, 0)
    mu_y = F.avg_pool2d(target, window_size, 1, 0)

    sigma_x = F.avg_pool2d(pred * pred, window_size, 1, 0) - mu_x ** 2
    sigma_y = F.avg_pool2d(target * target, window_size, 1, 0) - mu_y ** 2
    sigma_xy = F.avg_pool2d(pred * target, window_size, 1, 0) - mu_x * mu_y

    ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    return (ssim_n / (ssim_d + 1e-7)).mean()


def depth_loss(pred, target):
    l1 = torch.mean(torch.abs(pred - target))
    ssim_val = ssim(pred, target)
    return 0.85 * l1 + 0.15 * (1.0 - ssim_val)


# ===========================================================
#  Training / validation
# ===========================================================

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running = 0.0
    for rgb, depth in loader:
        rgb = rgb.to(device)
        depth = depth.to(device)

        optimizer.zero_grad()
        pred = model(rgb)
        loss = depth_loss(pred, depth)
        loss.backward()
        optimizer.step()

        running += loss.item() * rgb.size(0)
    return running / len(loader.dataset)


def validate_one_epoch(model, loader, device):
    model.eval()
    running = 0.0
    with torch.no_grad():
        for rgb, depth in loader:
            rgb = rgb.to(device)
            depth = depth.to(device)

            pred = model(rgb)
            loss = depth_loss(pred, depth)
            running += loss.item() * rgb.size(0)
    return running / len(loader.dataset)


# ===========================================================
#  Visualization helpers
# ===========================================================

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


def save_depth_examples(
    model,
    dataset,
    device,
    out_path,
    num_examples=10,
    seed=123,
    candidate_pool=80,
    min_rel_range=0.15
):
    """
    Save qualitative examples, excluding near-constant predictions.

    min_rel_range:
        Minimum (max-min) of prediction relative to GT range.
        0.15 = prediction must span at least 15% of GT depth variation.
    """
    rng = np.random.default_rng(seed)
    model.eval()

    candidate_pool = min(candidate_pool, len(dataset))
    candidate_indices = rng.choice(len(dataset), size=candidate_pool, replace=False)

    selected = []

    with torch.no_grad():
        for idx in candidate_indices:
            rgb, depth = dataset[idx]
            pred = model(rgb.unsqueeze(0).to(device)).cpu().squeeze(0)

            depth_np = depth.squeeze(0).numpy()
            pred_np = pred.squeeze(0).numpy()

            gt_range = depth_np.max() - depth_np.min()
            pred_range = pred_np.max() - pred_np.min()

            if gt_range > 1e-6 and (pred_range / gt_range) >= min_rel_range:
                selected.append(idx)

            if len(selected) == num_examples:
                break

    if len(selected) < num_examples:
        print(
            f"⚠️ Only found {len(selected)} informative predictions "
            f"(min_rel_range={min_rel_range})."
        )

    n = len(selected)
    if n == 0:
        print("⚠️ No valid examples found.")
        return

    plt.figure(figsize=(6, 2 * n))
    with torch.no_grad():
        for row, idx in enumerate(selected):
            rgb, depth = dataset[idx]
            pred = model(rgb.unsqueeze(0).to(device)).cpu().squeeze(0)

            depth_np = depth.squeeze(0).numpy()
            pred_np = pred.squeeze(0).numpy()

            vmin = float(depth_np.min())
            vmax = float(depth_np.max())

            ax = plt.subplot(n, 3, row * 3 + 1)
            ax.imshow(np.transpose(rgb.numpy(), (1, 2, 0)))
            ax.set_title(f"RGB (idx {idx})", fontsize=8)
            ax.axis("off")

            ax = plt.subplot(n, 3, row * 3 + 2)
            im = ax.imshow(depth_np, cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title("GT depth", fontsize=8)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.01)

            ax = plt.subplot(n, 3, row * 3 + 3)
            im = ax.imshow(pred_np, cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title("Pred depth", fontsize=8)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.01)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# ===========================================================
#  Main
# ===========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--usegeo_root", type=str, required=True,
                        help="Path to UseGeo root (folder containing Dataset-1, Dataset-2, Dataset-3)")
    parser.add_argument("--pretrained_path", type=str, required=True,
                        help="Path to best_model.pt from Part 1 (MidAir training)")
    parser.add_argument("--log_dir", type=str, default="runs/final_part2_usegeo")
    parser.add_argument("--outputs_dir", type=str, default="outputs_final_part2_usegeo")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.outputs_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------------------------------------
    # FIXED DATASET: deterministic split + separate objects
    # -------------------------------------------------------
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Build samples ONCE
    index_ds = UseGeoDepthDataset(args.usegeo_root, resize_hw=(192, 320), train=False)
    base_samples = index_ds.samples
    total_len = len(base_samples)

    val_len = int(0.2 * total_len)
    train_len = total_len - val_len

    # deterministic shuffle -> deterministic split
    idx = np.arange(total_len)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    train_idx = idx[:train_len]
    val_idx = idx[train_len:]

    # create two independent datasets (different transforms)
    train_dataset = UseGeoDepthDataset(args.usegeo_root, resize_hw=(192, 320), train=True)
    val_dataset = UseGeoDepthDataset(args.usegeo_root, resize_hw=(192, 320), train=False)

    # overwrite their samples so both use the same base sample list + split
    train_dataset.samples = [base_samples[i] for i in train_idx]
    val_dataset.samples = [base_samples[i] for i in val_idx]

    if args.max_train_samples is not None:
        train_dataset.samples = train_dataset.samples[:args.max_train_samples]
    if args.max_val_samples is not None:
        val_dataset.samples = val_dataset.samples[:args.max_val_samples]

    print(f"Total UseGeo samples: {total_len}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # -------- model --------
    model = DepthUNet(in_ch=3, out_ch=1, base_ch=32).to(device)

    print(f"Loading pretrained weights from {args.pretrained_path}")
    state_dict = torch.load(args.pretrained_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    # PRETRAINED BASELINE: save 10 prediction images BEFORE training
    print("\n===== Saving PRETRAINED baseline predictions on UseGeo (before training) =====")
    pretrained_examples_path = os.path.join(args.outputs_dir, "depth_predictions_usegeo_pretrained.png")
    save_depth_examples(model, val_dataset, device, pretrained_examples_path, num_examples=10, seed=123)
    print(f"Saved PRETRAINED depth prediction examples to {pretrained_examples_path}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    best_val = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== UseGeo Epoch {epoch}/{args.epochs} =====")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:03d} | train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f}")
        writer.add_scalar("Loss/train_usegeo", train_loss, epoch)
        writer.add_scalar("Loss/val_usegeo", val_loss, epoch)

        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(args.outputs_dir, "best_model_usegeo.pt")
            torch.save(model.state_dict(), best_path)
            print(f"  ✅ New best val loss. Model saved to {best_path}")

    writer.close()

    loss_plot_path = os.path.join(args.outputs_dir, "loss_curve_usegeo.png")
    save_loss_plot(train_losses, val_losses, loss_plot_path)
    print(f"Saved UseGeo loss curve to {loss_plot_path}")

    examples_path = os.path.join(args.outputs_dir, "depth_predictions_usegeo.png")
    save_depth_examples(model, val_dataset, device, examples_path, num_examples=10, seed=123)
    print(f"Saved UseGeo depth prediction examples to {examples_path}")


if __name__ == "__main__":
    main()