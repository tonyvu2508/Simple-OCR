"""
Data augmentation pipeline for OCR text recognition.

Provides document-specific augmentations beyond standard image transforms:
perspective distortion, Gaussian noise, JPEG artifacts, and controlled
brightness/contrast variations. Designed to simulate real-world degradation
of scanned auction documents.
"""

from __future__ import annotations

import random
from typing import Optional, Tuple

import cv2
import numpy as np


class OCRAugmentor:
    """
    Augmentation pipeline tailored for OCR text crops.
    
    Unlike generic image augmentation, this focuses on degradations that
    preserve text readability while teaching the model robustness:
    - Perspective distortion (skewed scans)
    - Gaussian noise (sensor noise)
    - Brightness/contrast variation (lighting conditions)
    - JPEG compression artifacts
    - Slight blur (out-of-focus or motion)
    
    Args:
        perspective_scale: Maximum perspective distortion magnitude.
        rotation_range: Maximum rotation in degrees.
        noise_var: Gaussian noise variance.
        brightness_range: Tuple of (min, max) brightness multiplier.
        contrast_range: Tuple of (min, max) contrast multiplier.
        blur_prob: Probability of applying blur.
        blur_kernel_range: Tuple of (min, max) blur kernel sizes.
        jpeg_quality_range: Tuple of (min, max) JPEG quality levels.
        augment_prob: Global probability of applying any augmentation.
    """

    def __init__(
        self,
        perspective_scale: float = 0.05,
        rotation_range: float = 3.0,
        noise_var: float = 0.01,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        blur_prob: float = 0.1,
        blur_kernel_range: Tuple[int, int] = (3, 5),
        jpeg_quality_range: Tuple[int, int] = (70, 95),
        augment_prob: float = 0.5,
    ):
        self.perspective_scale = perspective_scale
        self.rotation_range = rotation_range
        self.noise_var = noise_var
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.blur_prob = blur_prob
        self.blur_kernel_range = blur_kernel_range
        self.jpeg_quality_range = jpeg_quality_range
        self.augment_prob = augment_prob

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to an image.
        
        Args:
            image: Input image as numpy array (H, W, C), uint8 or float32.
        
        Returns:
            Augmented image with same dtype and shape.
        """
        if random.random() > self.augment_prob:
            return image
        
        original_dtype = image.dtype
        
        # Convert to float32 for processing
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # Apply augmentations in sequence (each has its own probability)
        if random.random() < 0.3:
            image = self._perspective_distortion(image)
        
        if random.random() < 0.3:
            image = self._rotate(image)
        
        if random.random() < 0.3:
            image = self._adjust_brightness_contrast(image)
        
        if random.random() < 0.2:
            image = self._add_gaussian_noise(image)
        
        if random.random() < self.blur_prob:
            image = self._apply_blur(image)
        
        if random.random() < 0.15:
            image = self._jpeg_compression(image)
        
        # Clip to valid range
        image = np.clip(image, 0.0, 1.0)
        
        # Convert back to original dtype
        if original_dtype == np.uint8:
            image = (image * 255).astype(np.uint8)
        
        return image

    def _perspective_distortion(self, image: np.ndarray) -> np.ndarray:
        """Apply random perspective distortion to simulate skewed scans."""
        h, w = image.shape[:2]
        
        # Generate random perspective offsets
        scale = self.perspective_scale
        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst_pts = src_pts + np.random.uniform(
            -scale * min(h, w), scale * min(h, w), src_pts.shape
        ).astype(np.float32)
        
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        result = cv2.warpPerspective(
            image, M, (w, h),
            borderMode=cv2.BORDER_REPLICATE
        )
        return result

    def _rotate(self, image: np.ndarray) -> np.ndarray:
        """Apply slight rotation to simulate imperfect scanning."""
        h, w = image.shape[:2]
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        result = cv2.warpAffine(
            image, M, (w, h),
            borderMode=cv2.BORDER_REPLICATE
        )
        return result

    def _adjust_brightness_contrast(self, image: np.ndarray) -> np.ndarray:
        """Randomly adjust brightness and contrast."""
        brightness = random.uniform(*self.brightness_range)
        contrast = random.uniform(*self.contrast_range)
        
        # Apply contrast around mean, then brightness
        mean = np.mean(image)
        result = (image - mean) * contrast + mean * brightness
        return result

    def _add_gaussian_noise(self, image: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to simulate sensor noise."""
        noise = np.random.normal(0, self.noise_var ** 0.5, image.shape).astype(np.float32)
        return image + noise

    def _apply_blur(self, image: np.ndarray) -> np.ndarray:
        """Apply slight Gaussian blur to simulate out-of-focus capture."""
        kernel_size = random.choice(
            range(self.blur_kernel_range[0], self.blur_kernel_range[1] + 1, 2)
        )
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def _jpeg_compression(self, image: np.ndarray) -> np.ndarray:
        """Simulate JPEG compression artifacts."""
        quality = random.randint(*self.jpeg_quality_range)
        
        # Convert to uint8 for JPEG encoding
        img_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        
        # Encode and decode with JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        _, encoded = cv2.imencode(".jpg", img_uint8, encode_params)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR if len(image.shape) == 3 else cv2.IMREAD_GRAYSCALE)
        
        # Convert back to float32
        return decoded.astype(np.float32) / 255.0


def letterbox_image(
    image: np.ndarray,
    target_height: int,
    target_width: int,
    pad_value: int = 0,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize image to target size while preserving aspect ratio using letterboxing.
    
    Pads the shorter dimension to reach the target size without distorting
    the text. This is critical for OCR — naive resize can destroy character
    shapes and make recognition impossible.
    
    Args:
        image: Input image (H, W, C) or (H, W).
        target_height: Target height.
        target_width: Target width.
        pad_value: Padding pixel value (0=black, 255=white).
    
    Returns:
        Tuple of (resized_image, scale_factor, (pad_top, pad_left)).
    """
    h, w = image.shape[:2]
    
    # Calculate scale to fit within target while preserving aspect ratio
    scale = min(target_height / h, target_width / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Calculate padding
    pad_top = (target_height - new_h) // 2
    pad_bottom = target_height - new_h - pad_top
    pad_left = (target_width - new_w) // 2
    pad_right = target_width - new_w - pad_left
    
    # Apply padding
    if len(image.shape) == 3:
        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(pad_value, pad_value, pad_value)
        )
    else:
        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=pad_value
        )
    
    return padded, scale, (pad_top, pad_left)


def deskew_image(image: np.ndarray, max_angle: float = 15.0) -> np.ndarray:
    """
    Detect and correct text skew in an image.
    
    Uses Hough line detection to estimate the dominant text angle,
    then rotates to correct it. Only corrects small angles to avoid
    false corrections on non-skewed text.
    
    Args:
        image: Input image (H, W, C) or (H, W).
        max_angle: Maximum skew angle to correct (degrees).
    
    Returns:
        Deskewed image.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Hough line detection
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100,
        minLineLength=50, maxLineGap=10
    )
    
    if lines is None:
        return image
    
    # Calculate angles of detected lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Only consider near-horizontal lines (text lines)
        if abs(angle) < max_angle:
            angles.append(angle)
    
    if not angles:
        return image
    
    # Use median angle for robustness against outliers
    median_angle = np.median(angles)
    
    # Only correct if angle is significant
    if abs(median_angle) < 0.5:
        return image
    
    # Rotate to correct skew
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    corrected = cv2.warpAffine(
        image, M, (w, h),
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return corrected
