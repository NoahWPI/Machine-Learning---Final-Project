Monocular Depth Estimation: MidAir → UseGeo Transfer
Overview
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
This project implements a monocular depth estimation pipeline using an encoder–decoder convolutional neural network (U-Net style).
The model is first trained on the synthetic MidAir dataset and then evaluated and fine-tuned on the real-world UseGeo dataset to study sim-to-real generalization.

The project is divided into two parts:

Part 1 (MidAir): Train a depth estimation model on synthetic aerial imagery

Part 2 (UseGeo): Evaluate and fine-tune the pretrained model on real UAV imagery and analyze domain shift

The repository contains fully reproducible training code, loss curves, and qualitative depth visualizations for both datasets.

Repository Structure:
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
├── Final_Project_Midair.py        # Part 1: Train on MidAir

├── Final_Project_UseGeo.py        # Part 2: Fine-tune & evaluate on UseGeo

├── outputs_final_part1/           # Auto-generated MidAir outputs

│   ├── best_model.pt

│   ├── loss_curve.png

│   └── depth_predictions_grid.png

├── outputs_final_part2_usegeo/    # Auto-generated UseGeo outputs

│   ├── best_model_usegeo.pt

│   ├── loss_curve_usegeo.png

│   ├── depth_predictions_usegeo_pretrained.png

│   └── depth_predictions_usegeo.png

└── README.md

Note: Datasets are not included in this repository due to size. See dataset setup below.

Environment & Dependencies
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
Tested with:

Python 3.9+

PyTorch

torchvision

numpy

matplotlib

pillow

tensorboard

Install dependencies using:
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
pip install torch torchvision numpy matplotlib pillow tensorboard

GPU is recommended but not required.

Dataset Setup
MidAir Dataset (Part 1)
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
Expected directory structure:

MidAir_root/

├── color_left/

│   └── trajectory_xxxx/frames/*.JPG

└── depth/

|   └── trajectory_xxxx/frames/*.PNG


Only a subset of MidAir is required (e.g., Kite_training/sunny).

UseGeo Dataset (Part 2)
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
Expected directory structure:

UseGeo_root/

├── Dataset-1/

│   ├── undistorted_images/

│   └── depth_maps/

├── Dataset-2/

│   ├── undistorted_images/

│   └── depth_maps/

└── Dataset-3/

|   ├── undistorted_images/

|   └── depth_maps/


Depth maps are normalized per-image to [0, 1] to handle scale inconsistencies in real data.

Part 1: Training on MidAir
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
Command

python Final_Project_Midair.py \

  --root_dir /path/to/MidAir_subset \
  
  --epochs 15 \
  
  --batch_size 4

What This Does
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
Trains a U-Net depth model from scratch on MidAir

Uses L1 + SSIM loss

Saves
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
Best model (best_model.pt)

Training/validation loss curve

10 qualitative depth predictions

Outputs
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
outputs_final_part1/

├── best_model.pt

├── loss_curve.png

└── depth_predictions_grid.png

Part 2: Sim-to-Real Transfer on UseGeo
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
Command

python Final_Project_UseGeo.py \

  --usegeo_root /path/to/UseGeo \
  
  --pretrained_path outputs_final_part1/best_model.pt \
  
  --epochs 15 \
  
  --batch_size 4

What This Does
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
Evaluates the MidAir-trained model on UseGeo (before training)

Fine-tunes the model on UseGeo

Re-evaluates after training

Saves
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
both pre- and post-training depth visualizations

Outputs
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
outputs_final_part2_usegeo/

├── best_model_usegeo.pt

├── loss_curve_usegeo.png

├── depth_predictions_usegeo_pretrained.png

└── depth_predictions_usegeo.png

Evaluation Strategy
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
Quantitative

Training vs. validation loss (L1 + SSIM)

Validation loss remaining slightly lower than training loss is expected and acceptable due to:

Data augmentation

Per-image normalization

Small dataset size

Qualitative

10 depth prediction examples per experiment

Uses consistent color scaling between GT and prediction

Filters out near-constant predictions to ensure meaningful visualization

Sim-to-Real Transfer Findings
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
Pretrained MidAir model performs poorly on UseGeo
→ Predictions collapse to low-variance depth maps

Fine-tuning on UseGeo recovers meaningful structure
→ Terrain gradients and scene geometry align well with ground truth

Demonstrates that synthetic pretraining alone is insufficient, but beneficial when combined with real-data fine-tuning

Notes on Reproducibility
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
Deterministic dataset splitting (fixed random seed)

Separate dataset objects for train/validation

Identical architecture and loss across datasets

All plots and visualizations generated automatically
