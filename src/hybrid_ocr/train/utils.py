"""
Training utilities: metrics calculation, logging, etc.
"""

from __future__ import annotations

import Levenshtein
from typing import List, Tuple


def calculate_metrics(
    predictions: List[str],
    targets: List[str],
) -> Tuple[float, float, float]:
    """
    Calculate OCR metrics.
    
    Args:
        predictions: List of predicted strings.
        targets: List of ground truth strings.
    
    Returns:
        Tuple of (accuracy, CER, WER).
        - accuracy: Exact match ratio (0-1).
        - CER: Character Error Rate.
        - WER: Word Error Rate (less meaningful for unsegmented Japanese, but included).
    """
    if not predictions or not targets or len(predictions) != len(targets):
        return 0.0, 0.0, 0.0
    
    exact_matches = 0
    total_char_dist = 0
    total_chars = 0
    total_word_dist = 0
    total_words = 0
    
    for pred, target in zip(predictions, targets):
        # Exact match
        if pred == target:
            exact_matches += 1
            
        # Character Error Rate (CER)
        dist = Levenshtein.distance(pred, target)
        total_char_dist += dist
        total_chars += max(len(target), 1)  # Avoid division by zero
        
        # Word Error Rate (WER) - simplistic split for Japanese
        # Real Japanese WER requires morphological analysis (e.g. MeCab)
        pred_words = list(pred)  # Treat chars as words for this approximation
        target_words = list(target)
        
        # Alternatively, if there are spaces, split by space
        if " " in target or " " in pred:
            pred_words = pred.split()
            target_words = target.split()
            
        w_dist = Levenshtein.distance(pred_words, target_words)
        total_word_dist += w_dist
        total_words += max(len(target_words), 1)
        
    accuracy = exact_matches / len(targets)
    cer = total_char_dist / total_chars
    wer = total_word_dist / total_words
    
    return accuracy, cer, wer


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
