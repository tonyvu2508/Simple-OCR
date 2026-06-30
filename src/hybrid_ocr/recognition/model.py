"""
Combined HybridOCR recognition model.

Combines ConvNeXt Visual Encoder and Transformer Decoder into a single
end-to-end recognition model. Supports both training (teacher forcing)
and inference (autoregressive decoding) modes.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .encoder import ConvNeXtEncoder
from .decoder import TransformerOCRDecoder
from ..dataset.vocabulary import Vocabulary


class HybridOCR(nn.Module):
    """
    Hybrid OCR Recognition Model: ConvNeXt Encoder + Transformer Decoder.
    
    This model takes a text crop image and produces a sequence of characters.
    
    Training mode (teacher forcing):
        image + target_input → logits
        Loss is computed between logits and target_output.
    
    Inference mode (autoregressive):
        image → predicted character sequence
    
    Architecture diagram:
        Image (B, 3, H, W)
            → ConvNeXt Encoder → visual tokens (B, seq_len, d_model)
            → Transformer Decoder:
                - Self-attention on generated characters
                - Cross-attention on visual tokens
            → Character logits (B, tgt_len, vocab_size)
    
    Args:
        vocab_size: Size of character vocabulary.
        d_model: Model dimension shared between encoder and decoder.
        nhead: Number of attention heads in decoder.
        num_decoder_layers: Number of transformer decoder layers.
        dim_feedforward: Feedforward hidden dimension in decoder.
        dropout: Dropout rate.
        max_seq_length: Maximum output sequence length.
        encoder_backbone: ConvNeXt variant ('tiny' or 'small').
        pretrained_encoder: Whether to use ImageNet pretrained encoder.
        freeze_encoder: Whether to freeze encoder weights initially.
        pad_idx: Padding token index.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 100,
        encoder_backbone: str = "tiny",
        pretrained_encoder: bool = True,
        freeze_encoder: bool = False,
        pad_idx: int = 0,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Phase 2: ConvNeXt Visual Encoder
        self.encoder = ConvNeXtEncoder(
            d_model=d_model,
            backbone=encoder_backbone,
            pretrained=pretrained_encoder,
            freeze_backbone=freeze_encoder,
        )
        
        # Phase 3: Transformer Decoder
        self.decoder = TransformerOCRDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_length=max_seq_length,
            pad_idx=pad_idx,
        )

    def forward(
        self,
        images: torch.Tensor,
        target_input: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for training with teacher forcing.
        
        Args:
            images: Batch of text crop images (B, 3, H, W).
            target_input: Decoder input tokens (B, tgt_len).
                          Format: [SOS, char1, char2, ...]
        
        Returns:
            Logits over vocabulary (B, tgt_len, vocab_size).
        """
        # Encode: image → visual token sequence
        memory = self.encoder(images)
        
        # Decode: visual tokens + target input → character logits
        logits = self.decoder(target_input, memory)
        
        return logits

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        vocab: Vocabulary,
        decoding: str = "greedy",
        beam_width: int = 5,
        max_len: Optional[int] = None,
    ) -> List[Dict[str, object]]:
        """
        Run inference on a batch of images.
        
        Args:
            images: Batch of text crop images (B, 3, H, W).
            vocab: Vocabulary instance for decoding indices to text.
            decoding: Decoding strategy ('greedy' or 'beam_search').
            beam_width: Beam width for beam search decoding.
            max_len: Maximum output sequence length.
        
        Returns:
            List of dictionaries, one per image:
                - "text": Predicted text string.
                - "tokens": Raw token indices.
                - "confidence": Per-character confidence scores.
        """
        self.eval()
        
        # Encode all images
        memory = self.encoder(images)
        
        results = []
        
        if decoding == "greedy":
            # Greedy decode the entire batch at once
            generated = self.decoder.greedy_decode(
                memory, vocab.sos_idx, vocab.eos_idx, max_len
            )
            
            # Also get confidence scores via a forward pass
            logits = self.decoder(generated[:, :-1], memory)
            probs = torch.softmax(logits, dim=-1)
            
            for i in range(images.size(0)):
                tokens = generated[i].cpu().tolist()
                text = vocab.decode(tokens)
                
                # Calculate per-character confidence
                char_probs = []
                for j, token_idx in enumerate(tokens[1:]):  # Skip SOS
                    if token_idx == vocab.eos_idx:
                        break
                    if j < probs.size(1):
                        char_probs.append(probs[i, j, token_idx].item())
                
                avg_confidence = sum(char_probs) / len(char_probs) if char_probs else 0.0
                
                results.append({
                    "text": text,
                    "tokens": tokens,
                    "confidence": avg_confidence,
                    "char_confidences": char_probs,
                })
        
        elif decoding == "beam_search":
            # Beam search: process one image at a time
            for i in range(images.size(0)):
                single_memory = memory[i:i+1]
                generated = self.decoder.beam_search_decode(
                    single_memory, vocab.sos_idx, vocab.eos_idx,
                    beam_width, max_len
                )
                
                tokens = generated[0].cpu().tolist()
                text = vocab.decode(tokens)
                
                results.append({
                    "text": text,
                    "tokens": tokens,
                    "confidence": 0.0,  # Beam search doesn't easily give per-token confidence
                    "char_confidences": [],
                })
        else:
            raise ValueError(f"Unknown decoding strategy: {decoding}")
        
        return results

    def save_checkpoint(self, path: str, extra_info: Optional[dict] = None) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: File path for the checkpoint.
            extra_info: Additional info to include (e.g., epoch, optimizer state).
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "model_config": {
                "vocab_size": self.vocab_size,
                "d_model": self.d_model,
                "max_seq_length": self.max_seq_length,
            },
        }
        if extra_info:
            checkpoint.update(extra_info)
        
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        vocab_size: Optional[int] = None,
        device: str = "cpu",
        **kwargs,
    ) -> "HybridOCR":
        """
        Load model from checkpoint.
        
        Args:
            path: Path to checkpoint file.
            vocab_size: Vocabulary size (overrides checkpoint config if provided).
            device: Device to load model onto.
            **kwargs: Additional arguments passed to model constructor.
        
        Returns:
            Loaded HybridOCR model.
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint.get("model_config", {})
        
        if vocab_size is not None:
            config["vocab_size"] = vocab_size
        
        config.update(kwargs)
        
        model = cls(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        
        return model


def build_model_from_config(config: dict, vocab: Vocabulary) -> HybridOCR:
    """
    Build HybridOCR model from a configuration dictionary.
    
    Args:
        config: Model configuration (from recognition.yaml).
        vocab: Vocabulary instance.
    
    Returns:
        Configured HybridOCR model.
    """
    encoder_cfg = config.get("encoder", {})
    decoder_cfg = config.get("decoder", {})
    
    model = HybridOCR(
        vocab_size=vocab.size,
        d_model=decoder_cfg.get("d_model", 512),
        nhead=decoder_cfg.get("nhead", 8),
        num_decoder_layers=decoder_cfg.get("num_decoder_layers", 6),
        dim_feedforward=decoder_cfg.get("dim_feedforward", 2048),
        dropout=decoder_cfg.get("dropout", 0.1),
        max_seq_length=decoder_cfg.get("max_seq_length", 100),
        encoder_backbone=encoder_cfg.get("backbone", "tiny"),
        pretrained_encoder=encoder_cfg.get("pretrained", True),
        freeze_encoder=encoder_cfg.get("freeze_backbone", False),
        pad_idx=vocab.pad_idx,
    )
    
    return model


if __name__ == "__main__":
    # Build vocabulary and model
    vocab = Vocabulary.build_japanese_auction_vocab()
    print(f"Vocabulary: {vocab}")
    
    model = HybridOCR(
        vocab_size=vocab.size,
        d_model=512,
        nhead=8,
        num_decoder_layers=6,
        pretrained_encoder=False,  # Skip download for testing
    )
    
    # Test training forward pass
    B = 2
    images = torch.randn(B, 3, 64, 256)
    target_input = torch.randint(0, vocab.size, (B, 10))
    
    logits = model(images, target_input)
    print(f"\nTraining forward pass:")
    print(f"  Images:       {images.shape}")
    print(f"  Target input: {target_input.shape}")
    print(f"  Logits:       {logits.shape}")
    
    # Test inference
    predictions = model.predict(images, vocab, decoding="greedy", max_len=20)
    for i, pred in enumerate(predictions):
        print(f"\nPrediction {i}: '{pred['text']}' (confidence: {pred['confidence']:.3f})")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
