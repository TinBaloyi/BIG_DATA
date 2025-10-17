"""
3D U-Net Architecture for BraTS Glioma Segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ConvBlock(nn.Module):
    """3D Convolutional block with BatchNorm and ReLU"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, padding: int = 1):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, 
                     padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size, 
                     padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    """Downsampling block"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
    
    def forward(self, x):
        skip = self.conv(x)
        x = self.pool(skip)
        return x, skip


class UpBlock(nn.Module):
    """Upsampling block"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.upconv = nn.ConvTranspose3d(
            in_channels, out_channels, 
            kernel_size=2, stride=2
        )
        self.conv = ConvBlock(in_channels, out_channels)
    
    def forward(self, x, skip):
        x = self.upconv(x)
        
        # Handle size mismatch (if any)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], 
                            mode='trilinear', align_corners=False)
        
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class UNet3D(nn.Module):
    """
    3D U-Net for BraTS Glioma Segmentation
    
    Args:
        in_channels: Number of input channels (4 for BraTS: T1, T1Gd, T2, FLAIR)
        num_classes: Number of output classes (5 for BraTS: background + 4 labels)
        base_channels: Number of channels in first layer
        depth: Depth of U-Net
    """
    
    def __init__(self, 
                 in_channels: int = 4,
                 num_classes: int = 5,
                 base_channels: int = 32,
                 depth: int = 4):
        super().__init__()
        
        self.depth = depth
        channels = [base_channels * (2 ** i) for i in range(depth + 1)]
        
        # Encoder (downsampling path)
        self.encoder = nn.ModuleList()
        for i in range(depth):
            in_ch = in_channels if i == 0 else channels[i]
            self.encoder.append(DownBlock(in_ch, channels[i + 1]))
        
        # Bottleneck
        self.bottleneck = ConvBlock(channels[depth], channels[depth])
        
        # Decoder (upsampling path)
        self.decoder = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.decoder.append(UpBlock(channels[i + 1], channels[i]))
        
        # Final classification layer
        self.final_conv = nn.Conv3d(channels[0], num_classes, kernel_size=1)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
        
        Returns:
            Output tensor of shape (B, num_classes, D, H, W)
        """
        # Encoder
        skips = []
        for down in self.encoder:
            x, skip = down(x)
            skips.append(skip)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        for i, up in enumerate(self.decoder):
            skip = skips[-(i + 1)]
            x = up(x, skip)
        
        # Final classification
        x = self.final_conv(x)
        return x


class AttentionGate(nn.Module):
    """Attention gate for Attention U-Net"""
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUpBlock(nn.Module):
    """Upsampling block with attention gate"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.upconv = nn.ConvTranspose3d(
            in_channels, out_channels, 
            kernel_size=2, stride=2
        )
        self.attention = AttentionGate(
            F_g=out_channels, 
            F_l=out_channels, 
            F_int=out_channels // 2
        )
        self.conv = ConvBlock(in_channels, out_channels)
    
    def forward(self, x, skip):
        x = self.upconv(x)
        
        # Apply attention gate
        skip = self.attention(x, skip)
        
        # Handle size mismatch
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], 
                            mode='trilinear', align_corners=False)
        
        # Concatenate
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class AttentionUNet3D(nn.Module):
    """
    3D Attention U-Net for BraTS Glioma Segmentation
    Incorporates attention gates to focus on relevant features
    """
    
    def __init__(self, 
                 in_channels: int = 4,
                 num_classes: int = 5,
                 base_channels: int = 32,
                 depth: int = 4):
        super().__init__()
        
        self.depth = depth
        channels = [base_channels * (2 ** i) for i in range(depth + 1)]
        
        # Encoder
        self.encoder = nn.ModuleList()
        for i in range(depth):
            in_ch = in_channels if i == 0 else channels[i]
            self.encoder.append(DownBlock(in_ch, channels[i + 1]))
        
        # Bottleneck
        self.bottleneck = ConvBlock(channels[depth], channels[depth])
        
        # Decoder with attention
        self.decoder = nn.ModuleList()
        for i in range(depth - 1, -1, -1):
            self.decoder.append(AttentionUpBlock(channels[i + 1], channels[i]))
        
        # Final classification
        self.final_conv = nn.Conv3d(channels[0], num_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        skips = []
        for down in self.encoder:
            x, skip = down(x)
            skips.append(skip)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        for i, up in enumerate(self.decoder):
            skip = skips[-(i + 1)]
            x = up(x, skip)
        
        # Final classification
        x = self.final_conv(x)
        return x


def get_model(model_name: str = 'unet3d', **kwargs):
    """
    Factory function to get model
    
    Args:
        model_name: Name of model ('unet3d' or 'attention_unet3d')
        **kwargs: Additional arguments for model
    
    Returns:
        Model instance
    """
    models = {
        'unet3d': UNet3D,
        'attention_unet3d': AttentionUNet3D
    }
    
    if model_name not in models:
        raise ValueError(f"Model {model_name} not found. Available: {list(models.keys())}")
    
    return models[model_name](**kwargs)


if __name__ == '__main__':
    # Test model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = get_model('unet3d', in_channels=4, num_classes=5, base_channels=32)
    model = model.to(device)
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 4, 128, 128, 128).to(device)
    
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")