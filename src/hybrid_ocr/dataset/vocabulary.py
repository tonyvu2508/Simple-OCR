"""
Vocabulary management for Japanese auction document OCR.

Handles character-to-index and index-to-character mapping for the full
character set needed: Hiragana, Katakana (full + half-width), common Kanji,
digits (full + half-width), Latin letters, and special symbols.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional


# --- Character Set Definitions ---

# Special tokens (must be first in vocabulary)
SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]

# Hiragana (83 characters including dakuten, handakuten, small kana)
HIRAGANA = (
    "ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞ"
    "ただちぢっつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽ"
    "まみむめもゃやゅゆょよらりるれろゎわゐゑをん"
    "ゔゝゞ"
)

# Full-width Katakana
KATAKANA_FULL = (
    "ァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾ"
    "タダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポ"
    "マミムメモャヤュユョヨラリルレロヮワヰヱヲンヴヵヶ"
    "ー・"  # Long vowel mark and middle dot
)

# Half-width Katakana (commonly used in auction documents)
KATAKANA_HALF = (
    "ｦｧｨｩｪｫｬｭｮｯ"
    "ｰｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿ"
    "ﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎ"
    "ﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ"
    "ﾞﾟ"  # Dakuten and handakuten marks
)

# Common Kanji for vehicle auction documents
# Organized by category for maintainability
KANJI_VEHICLE = (
    # Document structure
    "車両詳細出品票画像訂正特記事項"
    # Vehicle info
    "年式名型排気量燃料走行検外装色評価点"
    "内閲覧件開催中"
    # Auction
    "会場番号回発行員"
    "スタート流万円千"
    # Vehicle parts / condition
    "修復歴有無傷凹塗装交換跡割腐食"
    "前後左右上下"
    # Document fields
    "初度登録月日期限"
    "自家用商業軽貨客乗"
    "冷房暖形状態"
    # Common terms
    "保管付記入注意申告欄"
    "手数料込税別"
    "不具合内容等"
    # Colors
    "白黒赤青緑黄銀灰紺茶"
    # Numbers in Kanji
    "一二三四五六七八九十百"
    # Common vehicle-related
    "新中古本体価格落札相場"
    "距離程度良好可不動"
    # Additional common
    "大小全半長短高低"
    "電動力計器盤"
    "定通常"
    # Reiwa/Heisei era
    "令和平成昭"
)

# Full-width digits
DIGITS_FULL = "０１２３４５６７８９"

# Half-width digits
DIGITS_HALF = "0123456789"

# Latin letters
LATIN_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LATIN_LOWER = "abcdefghijklmnopqrstuvwxyz"

# Symbols commonly found in auction documents
SYMBOLS = (
    "/-.,;:()[]{}!?@#$%&*+=<>_~'\""
    "　"  # Full-width space
    " "   # Half-width space
    "。、・「」『』【】"  # Japanese punctuation
    "×○△□◇●▲■◆"  # Condition symbols
    "℃㎞㎝㎜"  # Units
    "→←↑↓"  # Arrows
)

# Full-width Latin
LATIN_FULL_UPPER = "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
LATIN_FULL_LOWER = "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ"


class Vocabulary:
    """
    Manages character vocabulary for OCR model.
    
    Maps characters to integer indices and vice versa.
    Supports saving/loading vocabulary to/from JSON files.
    
    Special tokens:
        PAD (0): Padding token for batch alignment
        SOS (1): Start of sequence — decoder input begins with this
        EOS (2): End of sequence — decoder stops when this is predicted
        UNK (3): Unknown character fallback
    
    Example:
        >>> vocab = Vocabulary.build_japanese_auction_vocab()
        >>> indices = vocab.encode("アクア")
        >>> text = vocab.decode(indices)
    """

    def __init__(self, characters: List[str]):
        """
        Initialize vocabulary from a list of unique characters.
        
        Special tokens (PAD, SOS, EOS, UNK) are automatically prepended.
        
        Args:
            characters: List of unique characters (without special tokens).
        """
        # Remove duplicates while preserving order
        seen = set()
        unique_chars = []
        for ch in characters:
            if ch not in seen:
                seen.add(ch)
                unique_chars.append(ch)
        
        # Build full token list: special tokens first, then characters
        self._tokens = SPECIAL_TOKENS + unique_chars
        
        # Build lookup dictionaries
        self._char_to_idx = {ch: idx for idx, ch in enumerate(self._tokens)}
        self._idx_to_char = {idx: ch for idx, ch in enumerate(self._tokens)}
        
        # Store special token indices for easy access
        self.pad_idx = self._char_to_idx["<PAD>"]
        self.sos_idx = self._char_to_idx["<SOS>"]
        self.eos_idx = self._char_to_idx["<EOS>"]
        self.unk_idx = self._char_to_idx["<UNK>"]

    @property
    def size(self) -> int:
        """Total vocabulary size including special tokens."""
        return len(self._tokens)

    @property
    def num_characters(self) -> int:
        """Number of actual characters (excluding special tokens)."""
        return len(self._tokens) - len(SPECIAL_TOKENS)

    def char_to_idx(self, char: str) -> int:
        """Convert a character to its index. Returns UNK index for unknown chars."""
        return self._char_to_idx.get(char, self.unk_idx)

    def idx_to_char(self, idx: int) -> str:
        """Convert an index to its character. Returns <UNK> for invalid indices."""
        return self._idx_to_char.get(idx, "<UNK>")

    def encode(self, text: str, add_sos: bool = True, add_eos: bool = True) -> List[int]:
        """
        Encode a text string into a list of token indices.
        
        Args:
            text: Input text string.
            add_sos: Whether to prepend SOS token.
            add_eos: Whether to append EOS token.
        
        Returns:
            List of integer indices.
        """
        indices = [self.char_to_idx(ch) for ch in text]
        if add_sos:
            indices = [self.sos_idx] + indices
        if add_eos:
            indices = indices + [self.eos_idx]
        return indices

    def decode(self, indices: List[int], strip_special: bool = True) -> str:
        """
        Decode a list of token indices into a text string.
        
        Args:
            indices: List of integer indices.
            strip_special: Whether to remove special tokens from output.
        
        Returns:
            Decoded text string.
        """
        chars = []
        for idx in indices:
            token = self.idx_to_char(idx)
            if strip_special:
                if token == "<EOS>":
                    break  # Stop at EOS
                if token in ("<PAD>", "<SOS>", "<UNK>"):
                    continue
            chars.append(token)
        return "".join(chars)

    def save(self, path: str) -> None:
        """Save vocabulary to a JSON file."""
        data = {
            "version": "1.0",
            "special_tokens": SPECIAL_TOKENS,
            "characters": self._tokens[len(SPECIAL_TOKENS):],
            "size": self.size,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        """Load vocabulary from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["characters"])

    @classmethod
    def build_japanese_auction_vocab(
        cls,
        include_hiragana: bool = True,
        include_katakana: bool = True,
        include_kanji: bool = True,
        include_digits: bool = True,
        include_latin: bool = True,
        include_symbols: bool = True,
        extra_kanji_path: Optional[str] = None,
    ) -> "Vocabulary":
        """
        Build a vocabulary specifically for Japanese auction document OCR.
        
        Args:
            include_hiragana: Include hiragana characters.
            include_katakana: Include katakana (full + half-width).
            include_kanji: Include common vehicle/auction kanji.
            include_digits: Include digits (full + half-width).
            include_latin: Include Latin letters (full + half-width).
            include_symbols: Include common symbols and punctuation.
            extra_kanji_path: Path to a text file with additional kanji characters.
        
        Returns:
            Vocabulary instance with all requested character sets.
        """
        characters = []
        
        if include_hiragana:
            characters.extend(list(HIRAGANA))
        
        if include_katakana:
            characters.extend(list(KATAKANA_FULL))
            characters.extend(list(KATAKANA_HALF))
        
        if include_kanji:
            characters.extend(list(KANJI_VEHICLE))
            # Load additional kanji from file if provided
            if extra_kanji_path and Path(extra_kanji_path).exists():
                with open(extra_kanji_path, "r", encoding="utf-8") as f:
                    extra = f.read().strip()
                    characters.extend(list(extra))
        
        if include_digits:
            characters.extend(list(DIGITS_HALF))
            characters.extend(list(DIGITS_FULL))
        
        if include_latin:
            characters.extend(list(LATIN_UPPER))
            characters.extend(list(LATIN_LOWER))
            characters.extend(list(LATIN_FULL_UPPER))
            characters.extend(list(LATIN_FULL_LOWER))
        
        if include_symbols:
            characters.extend(list(SYMBOLS))
        
        return cls(characters)

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return f"Vocabulary(size={self.size}, characters={self.num_characters})"

    def __contains__(self, char: str) -> bool:
        return char in self._char_to_idx


if __name__ == "__main__":
    # Build and display vocabulary stats
    vocab = Vocabulary.build_japanese_auction_vocab()
    print(f"Vocabulary: {vocab}")
    print(f"  Special tokens: {len(SPECIAL_TOKENS)}")
    print(f"  Total size: {vocab.size}")
    
    # Test encode/decode
    test_texts = ["アクア", "NHP10", "27千km", "3.5点", "28.0万円", "R02年"]
    for text in test_texts:
        encoded = vocab.encode(text)
        decoded = vocab.decode(encoded)
        print(f"  '{text}' → {encoded[:8]}... → '{decoded}'")
    
    # Save vocabulary
    vocab.save("configs/vocabulary.json")
    print(f"\nVocabulary saved to configs/vocabulary.json")
