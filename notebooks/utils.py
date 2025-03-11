import re
from typing import Dict


def process_hashtags(text: str) -> str:
    """Extract hashtags while preserving text."""
    if not isinstance(text, str):
        return ""
    clean_text = re.sub(r'#\w+', '', text).strip()
    hashtags = re.findall(r'#\w+', text)
    return f"{clean_text} {' '.join(hashtags)}" if hashtags else clean_text

def remove_special_character(text: str) -> str:
    """Remove special characters from text."""
    if not isinstance(text, str):
        return ""
    return re.sub(r'\W+', ' ', text)

def remove_url(text: str) -> str:
    """Remove URLs from text."""
    if not isinstance(text, str):
        return ""
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def process_slangs(text: str, slang_dict: Dict[str, str]) -> str:
    """Replace slangs with their meanings."""
    if not isinstance(text, str):
        return ""
    words = text.split()
    return " ".join(slang_dict.get(word.lower(), word) for word in words)

def data_cleaning(text: str, slang_dict: Dict[str,str]) -> str:
    """Clean text by applying all preprocessing steps."""
    if not isinstance(text, str):
        return ""
    try:
        text = process_hashtags(text)
        text = process_slangs(text, slang_dict)
        text = remove_special_character(text)
        text = remove_url(text)
        return text
    except Exception as e:
        print(f"Error in data_cleaning: {str(e)}")
        return text
