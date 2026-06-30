"""
ConvNeXt Visual Encoder for OCR.

Extracts rich visual features from text crop images using a pretrained
ConvNeXt-Tiny backbone. The feature maps are flattened into a sequence
of visual tokens that the Transformer Decoder can attend to via
cross-attention.

ConvNeXt uses 7×7 depthwise convolutions for a large receptive field,
which is critical for capturing character structure and spatial relationships.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
    from torchvision.models import convnext_small, ConvNeXt_Small_Weights
except ImportError:
    raise ImportError("torchvision >= 0.15.0 is required: pip install torchvision")


class ConvNeXtEncoder(nn.Module):
    """
    Visual Encoder using ConvNeXt backbone.
    
    Takes a text crop image and outputs a sequence of visual feature tokens
    that encode the spatial and visual information of the text.
    
    Architecture:
        Input image (B, 3, H, W)
            → ConvNeXt backbone (remove classification head)
            → Feature map (B, 768, H', W')
            → Flatten spatial dims → (B, H'*W', 768)
            → Linear projection → (B, seq_len, d_model)
    
    The output sequence is consumed by the Transformer Decoder via
    cross-attention, where each visual token represents a spatial patch
    of the input image.
    
    Args:
        d_model: Output feature dimension (must match decoder's d_model).
        backbone: Which ConvNeXt variant to use ('tiny' or 'small').
        pretrained: Whether to use ImageNet pretrained weights.
        freeze_backbone: Whether to freeze backbone weights during training.
    """

    def __init__(
        self,
        d_model: int = 512,
        backbone: str = "tiny",
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Load pretrained ConvNeXt backbone
        if backbone == "tiny":
            weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            model = convnext_tiny(weights=weights)
            self.feature_dim = 768  # ConvNeXt-Tiny output channels
        elif backbone == "small":
            weights = ConvNeXt_Small_Weights.DEFAULT if pretrained else None
            model = convnext_small(weights=weights)
            self.feature_dim = 768  # ConvNeXt-Small also outputs 768
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Use 'tiny' or 'small'.")
        
        # Extract feature layers (remove avgpool + classifier head)
        # ConvNeXt structure: features → avgpool → classifier
        self.features = model.features
        
        # Adaptive pooling to handle variable input sizes
        # Output: (B, feature_dim, 2, target_w)
        # We keep height small but preserve width for sequence modeling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, None))
        
        # Project from ConvNeXt feature dim to decoder d_model
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        
        # Optionally freeze backbone
        if freeze_backbone:
            self.freeze()

    def freeze(self) -> None:
        """Freeze backbone weights (useful for initial fine-tuning)."""
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze backbone weights for end-to-end training."""
        for param in self.features.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract visual features and convert to token sequence.
        
        Args:
            x: Input images (B, 3, H, W), normalized.
        
        Returns:
            Visual token sequence (B, seq_len, d_model).
            seq_len = H' * W' where H', W' are the spatial dims after ConvNeXt.
        """
        # Extract features through ConvNeXt backbone
        # Input: (B, 3, H, W) → Output: (B, 768, H', W')
        features = self.features(x)
        
        B, C, H, W = features.shape
        
        # Flatten spatial dimensions: (B, C, H, W) → (B, H*W, C)
        # Each spatial position becomes one visual token
        features = features.flatten(2)  # (B, C, H*W)
        features = features.permute(0, 2, 1)  # (B, H*W, C)
        
        # Project to decoder dimension: (B, seq_len, C) → (B, seq_len, d_model)
        features = self.projection(features)
        
        return features


if __name__ == "__main__":
    # Test the encoder
    encoder = ConvNeXtEncoder(d_model=512, backbone="tiny", pretrained=False)
    
    # Simulate a batch of text crop images
    # Typical input: 64px height, 256px width
    dummy_input = torch.randn(2, 3, 64, 256)
    output = encoder(dummy_input)
    
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"  → Batch size: {output.shape[0]}")
    print(f"  → Sequence length: {output.shape[1]}")
    print(f"  → Feature dimension: {output.shape[2]}")
    
    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
