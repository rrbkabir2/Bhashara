"""
Bhashara Text Utilities
Devanagari text normalizer, Hindi/English syllable counter, and length budgeter.
"""

import re
from typing import Tuple
from bhashara.config import HINDI_SYLLABLES_PER_SEC, ENGLISH_SYLLABLES_PER_SEC

# Number to Hindi word mapping for Devanagari expansion (0-20 sample + multiples)
HINDI_NUMBERS = {
    "0": "शून्य", "1": "एक", "2": "दो", "3": "तीन", "4": "चार",
    "5": "पांच", "6": "छह", "7": "सात", "8": "आठ", "9": "नौ",
    "10": "दस", "11": "ग्यारह", "12": "बारह", "13": "तेरह",
    "14": "चौदह", "15": "पंद्रह", "16": "सोलह", "17": "सत्रह",
    "18": "अठारह", "19": "उन्नीस", "20": "बीस", "50": "पचास", "100": "सौ"
}

# Common English acronyms / abbreviations expanded for clean Hindi TTS
ACRONYM_REPLACEMENTS = {
    "AI": "एआई",
    "OK": "ओके",
    "TV": "टीवी",
    "Dr.": "डॉक्टर",
    "Mr.": "मिस्टर",
    "Mrs.": "मिसिज़",
    "vs": "बनाम",
    "etc": "वगैरह",
}


def count_hindi_syllables(text: str) -> int:
    """
    Counts phonological syllables in Devanagari text.
    In Devanagari, each vowel (matra or independent vowel) or consonant
    without virama constitutes an acoustic nucleus/syllable.
    """
    if not text.strip():
        return 0
        
    # Vowels (Independent)
    vowels = set("अआइईउऊऋएऐओऔअंअः")
    # Matras (Dependent vowel signs)
    matras = set("ािीुूृेैोौॉॅंःँ")
    # Virama (halant - cancels vowel)
    virama = "्"
    # Consonants
    consonants = set(
        "कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसहक़ख़ग़ज़ड़ढ़फ़"
    )

    syllable_count = 0
    chars = list(text)
    n = len(chars)
    
    i = 0
    while i < n:
        c = chars[i]
        if c in vowels:
            syllable_count += 1
        elif c in consonants:
            # Check if followed by virama
            if i + 1 < n and chars[i + 1] == virama:
                # Combined conjunct; does not create a new syllable on its own
                i += 1
            else:
                syllable_count += 1
        i += 1

    return max(1, syllable_count)


def count_english_syllables(text: str) -> int:
    """
    Estimates English syllables using vowel cluster heuristics.
    """
    words = re.findall(r"[a-zA-Z]+", text.lower())
    if not words:
        return 0
        
    total_syllables = 0
    for word in words:
        # Remove trailing silent 'e'
        if word.endswith("e") and len(word) > 2 and not word.endswith("le"):
            word = word[:-1]
        # Count vowel clusters
        matches = re.findall(r"[aeiouy]+", word)
        count = len(matches)
        total_syllables += max(1, count)
        
    return max(1, total_syllables)


def count_syllables(text: str, lang: str = "hi") -> int:
    """Dispatches syllable counter based on language."""
    if lang == "hi" or any("\u0900" <= c <= "\u097F" for c in text):
        return count_hindi_syllables(text)
    return count_english_syllables(text)


def calculate_syllable_budget(duration_sec: float, lang: str = "hi") -> int:
    """
    Calculates the maximum permissible syllables for a line to fit within duration_sec
    without requiring unnatural audio speedup.
    """
    rate = HINDI_SYLLABLES_PER_SEC if lang == "hi" else ENGLISH_SYLLABLES_PER_SEC
    # Target 95% of duration slot to leave natural breathing pauses
    budget = int(duration_sec * rate * 0.95)
    return max(2, budget)


def normalize_devanagari_text(text: str) -> str:
    """
    Sanitizes Hindi text for TTS synthesis:
    - Expands numeric digits to Hindi word phonetics
    - Normalizes Latin acronyms (AI -> एआई)
    - Removes unpronounceable symbols and normalizes whitespace
    """
    cleaned = text.strip()
    
    # 1. Expand acronyms
    for eng, hin in ACRONYM_REPLACEMENTS.items():
        pattern = r"\b" + re.escape(eng) + r"\b"
        cleaned = re.sub(pattern, hin, cleaned, flags=re.IGNORECASE)

    # 2. Convert digits to Hindi words
    for num, word in sorted(HINDI_NUMBERS.items(), key=lambda x: -len(x[0])):
        pattern = r"\b" + re.escape(num) + r"\b"
        cleaned = re.sub(pattern, word, cleaned)

    # 3. Clean up punctuation artifacts (keep Hindi danda ।, period, comma, question mark)
    cleaned = re.sub(r"[#@$%^*_=~<>/\\|]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    
    return cleaned
