"""
Hybrid OCR: YOLO + ConvNeXt + Transformer Decoder
=================================================

A modern OCR system for Japanese auction vehicle documents.

Architecture:
    1. YOLO (Detection) → Detect and classify text regions
    2. ConvNeXt (Encoder) → Extract visual features from text crops
    3. Transformer (Decoder) → Decode feature sequences into text

Usage:
    from hybrid_ocr.pipeline import AuctionOCRPipeline
    
    pipeline = AuctionOCRPipeline()
    results = pipeline.process_pdf("path/to/document.pdf")
"""

__version__ = "0.1.0"
