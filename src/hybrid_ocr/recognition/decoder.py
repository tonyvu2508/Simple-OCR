"""
Transformer Decoder for OCR sequence recognition.

Decodes visual feature sequences from ConvNeXt into character sequences
using autoregressive generation. Implements the standard Transformer
Decoder architecture with:

- Self-Attention: models contextual relationships between generated characters
- Cross-Attention: links character predictions to visual features
- Positional Encoding: provides sequence position information
- Causal masking: prevents attending to future tokens during autoregressive decoding
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding.
    
    Injects information about the absolute position of tokens in the sequence.
    Uses the standard sin/cos formulation from "Attention Is All You Need":
    
        PE(pos, 2i)     = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i + 1) = cos(pos / 10000^(2i/d_model))
    
    This is preferred over learned positional encoding for OCR because
    it can generalize to sequence lengths not seen during training.
    
    Args:
        d_model: Model dimension.
        max_len: Maximum sequence length to precompute.
        dropout: Dropout rate applied after adding positional encoding.
    """

    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Precompute positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        # Register as buffer (not a parameter, but moves with model to device)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Input tensor (B, seq_len, d_model).
        
        Returns:
            Tensor with positional encoding added (B, seq_len, d_model).
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerOCRDecoder(nn.Module):
    """
    Transformer Decoder for OCR character sequence generation.
    
    Takes visual feature tokens from ConvNeXt and autoregressively generates
    a sequence of characters. During training, uses teacher forcing (feeds
    ground truth previous tokens). During inference, feeds its own predictions.
    
    Architecture:
        Character embedding + Positional Encoding
            → N × TransformerDecoderLayer:
                - Masked Self-Attention (causal — can't see future tokens)
                - Cross-Attention (attend to visual features)
                - Feed-forward network
            → Linear projection to vocabulary logits
    
    Args:
        vocab_size: Size of character vocabulary (including special tokens).
        d_model: Model dimension (must match encoder output).
        nhead: Number of attention heads.
        num_layers: Number of transformer decoder layers.
        dim_feedforward: Feedforward network hidden dimension.
        dropout: Dropout rate.
        max_seq_length: Maximum output sequence length.
        pad_idx: Padding token index (for masking).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_length: int = 100,
        pad_idx: int = 0,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.pad_idx = pad_idx
        
        # Character embedding: maps token indices to d_model vectors
        self.char_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_idx
        )
        
        # Positional encoding: injects sequence position information
        self.pos_encoding = PositionalEncoding(
            d_model, max_len=max_seq_length, dropout=dropout
        )
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (B, seq, features) format
            activation="gelu",
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )
        
        # Output projection: d_model → vocab_size logits
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform for better convergence."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate causal (look-ahead) mask for autoregressive decoding.
        
        Prevents the decoder from attending to future positions.
        Position i can only attend to positions <= i.
        
        Args:
            seq_len: Length of the target sequence.
            device: Device to create mask on.
        
        Returns:
            Causal mask tensor (seq_len, seq_len) where True means "blocked".
        """
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )
        return mask

    def _generate_padding_mask(
        self, tgt: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Generate padding mask for target sequence.
        
        Masks out padding tokens so they don't participate in attention.
        
        Args:
            tgt: Target token indices (B, seq_len).
        
        Returns:
            Padding mask (B, seq_len) where True means "is padding, ignore it".
        """
        return tgt == self.pad_idx

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the decoder.
        
        Used during training with teacher forcing: tgt contains ground truth
        tokens shifted right (SOS prepended, last token removed).
        
        Args:
            tgt: Target token indices (B, tgt_seq_len).
                 During training: [SOS, char1, char2, ...]
            memory: Encoder output — visual feature tokens (B, src_seq_len, d_model).
            tgt_mask: Optional causal mask (auto-generated if None).
            tgt_key_padding_mask: Optional padding mask for target.
        
        Returns:
            Logits over vocabulary for each position (B, tgt_seq_len, vocab_size).
        """
        seq_len = tgt.size(1)
        
        # Generate causal mask if not provided
        if tgt_mask is None:
            tgt_mask = self._generate_causal_mask(seq_len, tgt.device)
        
        # Generate padding mask if not provided
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = self._generate_padding_mask(tgt)
        
        # Embed target tokens and add positional encoding
        # (B, seq_len) → (B, seq_len, d_model)
        tgt_emb = self.char_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb)
        
        # Run through transformer decoder
        # Cross-attention happens inside: tgt_emb attends to memory (visual features)
        output = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        
        # Project to vocabulary logits
        logits = self.output_projection(output)
        
        return logits

    @torch.no_grad()
    def greedy_decode(
        self,
        memory: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        max_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Greedy autoregressive decoding (inference mode).
        
        Starts with SOS token and generates one character at a time,
        always choosing the highest probability character. Stops when
        EOS is predicted or max length is reached.
        
        Args:
            memory: Encoder output (B, src_seq_len, d_model).
            sos_idx: Index of the SOS (start of sequence) token.
            eos_idx: Index of the EOS (end of sequence) token.
            max_len: Maximum output length (defaults to self.max_seq_length).
        
        Returns:
            Generated token indices (B, generated_len).
        """
        if max_len is None:
            max_len = self.max_seq_length
        
        B = memory.size(0)
        device = memory.device
        
        # Start with SOS token for each sample in batch
        generated = torch.full((B, 1), sos_idx, dtype=torch.long, device=device)
        
        for _ in range(max_len - 1):
            # Forward pass with current generated sequence
            logits = self.forward(generated, memory)
            
            # Take the last position's prediction
            next_token_logits = logits[:, -1, :]  # (B, vocab_size)
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # (B, 1)
            
            # Append predicted token
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check if all sequences have generated EOS
            if (next_token == eos_idx).all():
                break
        
        return generated

    @torch.no_grad()
    def beam_search_decode(
        self,
        memory: torch.Tensor,
        sos_idx: int,
        eos_idx: int,
        beam_width: int = 5,
        max_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Beam search decoding for improved accuracy.
        
        Maintains top-k candidate sequences at each step, exploring
        multiple paths through the output space. More accurate than
        greedy decoding but proportionally slower.
        
        Args:
            memory: Encoder output (B, src_seq_len, d_model).
                    NOTE: Currently only supports B=1 for beam search.
            sos_idx: Index of the SOS token.
            eos_idx: Index of the EOS token.
            beam_width: Number of beams (candidate sequences) to maintain.
            max_len: Maximum output length.
        
        Returns:
            Best generated token indices (1, generated_len).
        """
        if max_len is None:
            max_len = self.max_seq_length
        
        assert memory.size(0) == 1, "Beam search currently supports batch_size=1 only"
        
        device = memory.device
        
        # Expand memory for beam_width beams
        memory_expanded = memory.expand(beam_width, -1, -1)
        
        # Initialize beams: each beam starts with SOS
        beams = torch.full((beam_width, 1), sos_idx, dtype=torch.long, device=device)
        beam_scores = torch.zeros(beam_width, device=device)
        beam_scores[1:] = -float("inf")  # Only first beam is active initially
        
        completed_beams = []
        completed_scores = []
        
        for step in range(max_len - 1):
            # Forward pass for all active beams
            logits = self.forward(beams, memory_expanded)
            next_logits = logits[:, -1, :]  # (beam_width, vocab_size)
            log_probs = torch.log_softmax(next_logits, dim=-1)
            
            # Calculate scores for all possible next tokens
            # (beam_width, vocab_size)
            next_scores = beam_scores.unsqueeze(-1) + log_probs
            
            # Flatten and select top-k
            vocab_size = next_scores.size(-1)
            next_scores_flat = next_scores.view(-1)
            topk_scores, topk_indices = next_scores_flat.topk(beam_width)
            
            # Determine which beam and which token each top-k came from
            beam_indices = topk_indices // vocab_size
            token_indices = topk_indices % vocab_size
            
            # Update beams
            new_beams = torch.cat([
                beams[beam_indices],
                token_indices.unsqueeze(-1)
            ], dim=1)
            beam_scores = topk_scores
            
            # Check for completed beams (predicted EOS)
            for i in range(beam_width):
                if token_indices[i] == eos_idx:
                    completed_beams.append(new_beams[i])
                    completed_scores.append(beam_scores[i].item())
            
            # Remove completed beams and continue
            active_mask = token_indices != eos_idx
            if not active_mask.any():
                break
            
            beams = new_beams[active_mask]
            beam_scores = beam_scores[active_mask]
            
            # Pad back to beam_width if needed
            if beams.size(0) < beam_width:
                pad_count = beam_width - beams.size(0)
                beams = torch.cat([beams, beams[:pad_count].clone()], dim=0)
                beam_scores = torch.cat([
                    beam_scores,
                    torch.full((pad_count,), -float("inf"), device=device)
                ])
            
            beams = beams[:beam_width]
            beam_scores = beam_scores[:beam_width]
        
        # Return best completed beam, or best active beam if none completed
        if completed_beams:
            best_idx = max(range(len(completed_scores)), key=lambda i: completed_scores[i])
            return completed_beams[best_idx].unsqueeze(0)
        else:
            return beams[0].unsqueeze(0)


if __name__ == "__main__":
    # Test the decoder
    vocab_size = 500
    d_model = 512
    
    decoder = TransformerOCRDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=8,
        num_layers=6,
    )
    
    # Simulate encoder output (visual tokens)
    B, src_seq_len = 2, 16
    memory = torch.randn(B, src_seq_len, d_model)
    
    # Simulate teacher forcing input
    tgt_seq_len = 10
    tgt = torch.randint(0, vocab_size, (B, tgt_seq_len))
    
    # Forward pass
    logits = decoder(tgt, memory)
    print(f"Decoder forward pass:")
    print(f"  Target input:  {tgt.shape}")
    print(f"  Memory input:  {memory.shape}")
    print(f"  Output logits: {logits.shape}")
    
    # Greedy decode
    generated = decoder.greedy_decode(memory, sos_idx=1, eos_idx=2, max_len=20)
    print(f"\nGreedy decode output: {generated.shape}")
    
    # Parameter count
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"\nTotal parameters: {total_params:,}")
