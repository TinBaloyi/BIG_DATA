# src/model.py
from monai.networks.nets import UNet
import torch.nn as nn

def build_unet3d(in_channels=4, num_classes=5, base_channels=(32,64,128,256,512)):
    """
    Simple MONAI UNet 3D builder.
    """
    model = UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=num_classes,
        channels=tuple(base_channels),
        strides=(2,2,2,2),
        num_res_units=2,
        norm="INSTANCE",
    )
    return model
