"""
Synthetic data generator for Japanese OCR training.

Renders text with various Japanese fonts onto images to create synthetic
training data. This is critical for pre-training the Transformer Decoder
which is extremely data-hungry — real annotated data alone is insufficient.

Generates images matching typical auction document text styles:
- Machine-printed text (Gothic, Mincho fonts)
- Form field values
- Mixed Japanese/Latin/numeric text
"""

from __future__ import annotations

import json
import os
import random
import string
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    raise ImportError("Pillow is required: pip install Pillow")

from .vocabulary import (
    DIGITS_HALF, DIGITS_FULL, HIRAGANA, KATAKANA_FULL, KATAKANA_HALF,
    KANJI_VEHICLE, LATIN_UPPER, LATIN_LOWER, SYMBOLS, Vocabulary,
)


# --- Text Templates for Auction Documents ---

# Typical patterns found in auction documents
TEXT_TEMPLATES = {
    "car_name": [
        "アクア", "プリウス", "ヴェルファイア", "アルファード", "ハリアー",
        "ノア", "ヴォクシー", "セレナ", "フィット", "ヴェゼル",
        "エクストレイル", "フォレスター", "インプレッサ", "レヴォーグ",
        "クラウン", "カムリ", "マークX", "レクサス", "スカイライン",
        "フェアレディZ", "ランドクルーザー", "パジェロ", "ジムニー",
        "ワゴンR", "タント", "N-BOX", "スペーシア", "ムーヴ",
        "デイズ", "ルークス", "ハスラー", "キャスト", "コペン",
    ],
    "model_code": [
        "NHP10", "ZVW30", "AGH30W", "GGH35W", "ZSU60W",
        "ZRR80G", "C27", "GK3", "RU1", "T32",
        "SJ5", "GP7", "VM4", "GRS210", "AXVH70",
        "DAA-NHP10", "3BA-AGH30W", "DBA-ZRR80G",
        "5AA-ZVW30", "6AA-AXVH70",
    ],
    "mileage": [
        "27千km", "38千km", "185千km", "5千km", "92千km",
        "12千km", "45千km", "78千km", "134千km", "3千km",
        "15万km", "2万km",
    ],
    "price": [
        "28.0万円", "53.8万円", "298.0万円", "355.6万円", "20.0万円",
        "70.7万円", "150.0万円", "88.5万円", "42.0万円", "199.8万円",
    ],
    "year": [
        "R02年", "R05年", "H28年", "R01年", "H30年",
        "R03年", "R04年", "H29年", "H27年", "R06年",
        "令和2年", "令和5年", "平成28年",
    ],
    "color": [
        "アオ", "シロ", "クロ", "パール", "シルバー",
        "ガンメタ", "ワイン", "グレー", "ゴールド", "ブラウン",
    ],
    "engine": [
        "1500cc/GS/AT", "2500cc/GS/DAT", "660cc/GS/CVT",
        "2000cc/DS/AT", "3000cc/GS/AT", "1800cc/GS/CVT",
        "1300cc/GS/MT", "2400cc/DS/DAT", "1200cc/GS/CVT",
    ],
    "rating": [
        "3.5点", "4点", "4.5点", "3点", "5点",
        "2.5点", "S点", "R点",
    ],
    "chassis": [
        "NHP10-6863568", "AGH30-0463625", "ZVW30-1234567",
        "GK3-9876543", "ZRR80-0112233",
    ],
    "form_labels": [
        "車名・グレード", "型式", "排気量", "走行", "車検",
        "出品番号", "初度登録年月", "形状", "燃料", "シフト",
        "色", "色コード", "修復歴", "セールスポイント",
        "注意事項申告欄", "車台番号", "登録番号", "R券",
        "検査員", "記入欄", "名変通知期限",
    ],
    "features": [
        "純正ナビ・TV・バックカメラ",
        "トヨタセーフティセンス・VSC",
        "PKSB・クリアランスソナー",
        "プッシュスタート・スマートキー",
        "レザーシート",
        "ETC",
        "ドライブレコーダー",
        "パワースライドドア",
        "サンルーフ",
    ],
}


class SyntheticDataGenerator:
    """
    Generate synthetic OCR training data with Japanese text.
    
    Creates text crop images by rendering random text combinations using
    various fonts. Simulates the look of both machine-printed and form-style
    text found in auction documents.
    
    Args:
        vocab: Vocabulary instance (generated text will only use chars in vocab).
        output_dir: Directory to save generated images and annotations.
        fonts_dir: Directory containing TTF/OTF font files.
                   If None, uses system default font.
        img_height: Output image height.
        img_width: Output image width.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        output_dir: str,
        fonts_dir: Optional[str] = None,
        img_height: int = 64,
        img_width: int = 256,
    ):
        self.vocab = vocab
        self.output_dir = Path(output_dir)
        self.img_height = img_height
        self.img_width = img_width
        
        # Load available fonts
        self.fonts = self._load_fonts(fonts_dir)
        
        # Create output directories
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def _load_fonts(self, fonts_dir: Optional[str]) -> List[str]:
        """Load font file paths from directory."""
        fonts = []
        
        if fonts_dir and Path(fonts_dir).exists():
            for ext in ("*.ttf", "*.otf", "*.ttc"):
                fonts.extend(str(p) for p in Path(fonts_dir).glob(ext))
        
        # Fallback: try common macOS and Linux system font paths for Japanese
        if not fonts:
            system_font_paths = [
                # macOS fallbacks
                "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
                "/System/Library/Fonts/ヒラギノ明朝 ProN.ttc",
                "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
                "/Library/Fonts/Arial Unicode.ttf",
                "/System/Library/Fonts/Hiragino Sans GB.ttc",
                # Linux/Ubuntu fallbacks (Takao, IPA, Noto CJK)
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/takao-gothic/TakaoGothic.ttf",
                "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
                "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
                "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
            ]
            for font_path in system_font_paths:
                if Path(font_path).exists():
                    fonts.append(font_path)
        
        if not fonts:
            print("WARNING: No Japanese fonts found. Text rendering may fail.")
            print("  Consider installing Japanese fonts or providing a fonts_dir.")
            fonts = [None]  # Will use PIL default font
        
        return fonts

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Get a random font at the specified size."""
        font_path = random.choice(self.fonts)
        
        if font_path is None:
            return ImageFont.load_default()
        
        try:
            return ImageFont.truetype(font_path, size)
        except (OSError, IOError):
            return ImageFont.load_default()

    def _generate_random_text(self) -> str:
        """Generate random text matching auction document patterns."""
        category = random.choice(list(TEXT_TEMPLATES.keys()))
        
        # 70% chance: use a template
        if random.random() < 0.7:
            text = random.choice(TEXT_TEMPLATES[category])
        else:
            # 30% chance: generate random combination
            text = self._generate_random_chars(random.randint(2, 15))
        
        return text

    def _generate_random_chars(self, length: int) -> str:
        """Generate a random string from vocabulary characters."""
        char_pools = []
        
        # Build weighted character pool
        char_pools.extend(list(DIGITS_HALF) * 3)      # Numbers are common
        char_pools.extend(list(KATAKANA_FULL) * 2)     # Katakana is common
        char_pools.extend(list(HIRAGANA))              # Hiragana
        char_pools.extend(list(KANJI_VEHICLE))          # Kanji
        char_pools.extend(list(LATIN_UPPER) * 2)       # Latin uppercase
        char_pools.extend(list(LATIN_LOWER))           # Latin lowercase
        
        # Filter to only characters in vocabulary
        valid_chars = [ch for ch in char_pools if ch in self.vocab]
        
        if not valid_chars:
            return "テスト"  # Fallback
        
        return "".join(random.choice(valid_chars) for _ in range(length))

    def _render_text_image(
        self,
        text: str,
        font_size_range: Tuple[int, int] = (20, 48),
    ) -> np.ndarray:
        """
        Render text onto a white background image.
        
        Args:
            text: Text to render.
            font_size_range: Range of font sizes to randomly select from.
        
        Returns:
            Rendered image as numpy array (H, W, 3), uint8.
        """
        font_size = random.randint(*font_size_range)
        font = self._get_font(font_size)
        
        # Create temporary image to measure text size
        tmp_img = Image.new("RGB", (1, 1), color=(255, 255, 255))
        tmp_draw = ImageDraw.Draw(tmp_img)
        
        try:
            bbox = tmp_draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except AttributeError:
            # Fallback for older Pillow versions
            text_w, text_h = tmp_draw.textsize(text, font=font)
        
        # Create image with padding
        pad_x = random.randint(5, 20)
        pad_y = random.randint(5, 15)
        img_w = text_w + 2 * pad_x
        img_h = text_h + 2 * pad_y
        
        # Ensure minimum size
        img_w = max(img_w, 32)
        img_h = max(img_h, 16)
        
        # Random background color (mostly white/light, sometimes dark)
        if random.random() < 0.85:
            # Light background
            bg_val = random.randint(220, 255)
            bg_color = (bg_val, bg_val, bg_val)
            text_val = random.randint(0, 50)
            text_color = (text_val, text_val, text_val)
        else:
            # Dark background (inverted)
            bg_val = random.randint(0, 50)
            bg_color = (bg_val, bg_val, bg_val)
            text_val = random.randint(200, 255)
            text_color = (text_val, text_val, text_val)
        
        # Create image and draw text
        img = Image.new("RGB", (img_w, img_h), color=bg_color)
        draw = ImageDraw.Draw(img)
        
        # Center text with slight random offset
        x_offset = pad_x + random.randint(-3, 3)
        y_offset = pad_y + random.randint(-3, 3)
        draw.text((x_offset, y_offset), text, font=font, fill=text_color)
        
        return np.array(img)

    def generate(
        self,
        num_samples: int = 10000,
        font_size_range: Tuple[int, int] = (20, 48),
    ) -> str:
        """
        Generate synthetic training dataset.
        
        Args:
            num_samples: Number of samples to generate.
            font_size_range: Range of font sizes.
        
        Returns:
            Path to the generated annotation JSON file.
        """
        annotations = []
        
        for i in range(num_samples):
            text = self._generate_random_text()
            
            # Skip empty or single-char texts occasionally
            if len(text) < 1:
                continue
            
            # Render text image
            image = self._render_text_image(text, font_size_range)
            
            # Save image
            img_filename = f"synth_{i:06d}.png"
            img_path = self.images_dir / img_filename
            cv2.imwrite(str(img_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            annotations.append({
                "image": f"images/{img_filename}",
                "label": text,
            })
            
            if (i + 1) % 1000 == 0:
                print(f"  Generated {i + 1}/{num_samples} samples")
        
        # Save annotations
        ann_path = self.output_dir / "annotations.json"
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)
        
        print(f"Generated {len(annotations)} synthetic samples")
        print(f"  Images: {self.images_dir}")
        print(f"  Annotations: {ann_path}")
        
        return str(ann_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic Japanese OCR training data")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--output-dir", default="data/synthetic_test", help="Output directory")
    args = parser.parse_args()
    
    vocab = Vocabulary.build_japanese_auction_vocab()
    
    generator = SyntheticDataGenerator(
        vocab=vocab,
        output_dir=args.output_dir,
        img_height=64,
        img_width=256,
    )
    
    ann_path = generator.generate(num_samples=args.num_samples)
    print(f"\nAnnotation file: {ann_path}")
