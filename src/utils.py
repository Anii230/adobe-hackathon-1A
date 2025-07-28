# utils.py

import re
import os
import spacy
from pathlib import Path
from langdetect import detect, DetectorFactory

# --- Constants and Global State ---

# Seed for langdetect to ensure consistent results
DetectorFactory.seed = 0

# Define input and output directories
INPUT_DIR = Path("data/input")
OUTPUT_DIR = Path("data/output")

# Global variables to hold the loaded spaCy model and detected language
NLP = None
DETECTED_LANG = None

# --- Functions ---

def load_spacy_model(lang: str):
    """Loads the appropriate spaCy model based on the detected language."""
    global NLP
    if lang == 'en':
        try:
            NLP = spacy.load("en_core_web_sm")
            print("✔️ Loaded spaCy model: 'en_core_web_sm'")
        except OSError:
            print("❌ spaCy model 'en_core_web_sm' not found. Please run:")
            print("    python -m spacy download en_core_web_sm")
            NLP = None
    elif lang == 'ja':
        try:
            NLP = spacy.load("ja_core_news_sm")
            print("✔️ Loaded spaCy model: 'ja_core_news_sm'")
        except OSError:
            print("❌ spaCy model 'ja_core_news_sm' not found. Please run:")
            print("    python -m spacy download ja_core_news_sm")
            NLP = None
    else:
        print(f"⚠️ No spaCy model configured for language: '{lang}'. NLP features will be disabled.")
        NLP = None

def clean_text(text: str) -> str:
    """Cleans a string by removing URLs, extra whitespace, and repeated words."""
    text = re.sub(r'https?://\S+|www\.\S+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    if DETECTED_LANG == 'en':
        text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text, flags=re.IGNORECASE)
    return text

def is_title_case(text: str) -> bool:
    """Checks if a string is in title case (for English)."""
    if DETECTED_LANG == 'ja': return False
    # Stop words to ignore when checking for title case
    stop_words = {'a', 'an', 'the', 'and', 'but', 'or', 'for', 'of', 'in', 'on', 'to'}
    words = [w for w in text.split() if w.isalpha() and w.lower() not in stop_words]
    if not words: return False
    return sum(1 for w in words if w[0].isupper()) / len(words) > 0.6

def roman_to_int(s: str) -> int:
    """Converts a Roman numeral string to an integer."""
    roman_map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    s = s.upper()
    val = 0
    try:
        for i in range(len(s)):
            if i > 0 and roman_map[s[i]] > roman_map[s[i - 1]]:
                val += roman_map[s[i]] - 2 * roman_map[s[i - 1]]
            else:
                val += roman_map[s[i]]
    except KeyError:
        return 0  # Not a valid Roman numeral
    return val

def normalize_text_for_comparison(text: str) -> str:
    """Normalizes text for deduplication by making it lowercase and removing non-essentials."""
    # Keeps alphanumeric characters and Japanese characters
    pattern = r'[^a-z0-9\s\u3000-\u303F\u3040-\u309F\u30A0-\u30FF\uFF00-\uFFEF\u4E00-\u9FAF]'
    return re.sub(r'\s+', ' ', re.sub(pattern, '', text.lower())).strip()
