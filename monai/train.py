import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path
import monai
from monai.networks.nets import SwinUNETR

from monai.losses import DiceLoss

# from monai.metrics import MeanSquaredError
from monai.data import Dataset, DataLoader, pad_list_data_collate
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityd,
    RandCropByPosNegLabeld,
    RandAffined,
    Spacingd,
    Orientationd,
    ConcatItemsd,
    ToTensord,
    Compose,
)

import numpy as np

# Define paths to your data folders
data_dir = Path(
    "/mnt/nfs/slow_ai_team/organ_segmentation/nnunet_liverv0.0/nnUNet_raw_database/nnUNet_raw/nnUNet_raw_data/Dataset019_AutoPET2024/"
)
images_dir = data_dir / "imagesTr"
labels_dir = data_dir / "labelsTr"

# Collect pairs of input images and their corresponding labels
image_files = sorted(images_dir.glob("*_0000.nii.gz"))

data_dicts = [
    {
        "CT": str(img),
        "PT": str(img).replace("_0000.nii.gz", "_0001.nii.gz"),
        "label": labels_dir / str(img.name).replace("_0000.nii.gz", ".nii.gz"),
    }
    for img in (image_files[0:100])
]

# Define transform operations
train_transforms = Compose(
    [
        LoadImaged(keys=["CT", "PT", "label"]),
        EnsureChannelFirstd(keys=["CT", "PT", "label"]),
        Spacingd(
            keys=["CT", "PT", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "bilinear", "nearest"),
        ),
        Orientationd(keys=["CT", "PT", "label"], axcodes="RAS"),
        ScaleIntensityd(keys=["CT", "PT"]),
        ConcatItemsd(
            ["CT", "PT"],
            "image",
        ),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(64, 64, 64),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
            allow_smaller=False,
        ),
        RandAffined(
            keys=["image", "label"],
            mode=("bilinear", "nearest"),
            prob=0.5,
            rotate_range=(0, 0, np.pi / 15),
            scale_range=(0.5, 0.5, 1),
        ),
        ToTensord(keys=["image", "label"]),
    ]
)

# Load dataset and apply transforms
train_ds = Dataset(data=data_dicts, transform=train_transforms)

# Create data loader
train_loader = DataLoader(
    train_ds,
    batch_size=1,
    shuffle=True,
    num_workers=2,
    collate_fn=pad_list_data_collate,
)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SwinUNETRModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SwinUNETRModel, self).__init__()
        self.swin_unetr = SwinUNETR(
            img_size=(64, 64, 64),  # Update this based on your data dimensions
            in_channels=in_channels,
            out_channels=out_channels,
            depths=(2, 2, 2, 2),
            num_heads=(4, 8, 16, 32),
        )

        # Modify the final layer to have no activation
        # num_features = self.swin_unetr.num_features
        self.final_conv = nn.Conv3d(1, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.swin_unetr(x)
        return self.final_conv(x)


# Instantiate the custom model
model = SwinUNETRModel(
    in_channels=2,  # Two input channels for *_0000.nii.gz and *_0001.nii.gz
    out_channels=1,  # Assuming single output for regression
).to(device)

# Define loss function and optimizer (changed loss function)
loss_function = nn.MSELoss()  # Changed to nn.MSELoss for Mean Squared Error
optimizer = torch.optim.Adam(model.parameters(), 1e-4)

# Training loop
max_epochs = 1000
for epoch in range(max_epochs):
    print(f"Epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0.0

    for batch_data in train_loader:
        inputs = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= len(train_loader)
    print(f"Loss: {epoch_loss}")

# Save trained model
torch.save(model.state_dict(), "swin_unetr_segmentation_model.pth")
