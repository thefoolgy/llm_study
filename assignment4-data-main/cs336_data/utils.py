from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.parse.encoding import detect_encoding
from resiliparse.extract.html2text import extract_plain_text
from typing import Tuple, Any
import fasttext
import re

def extract_text(html_bytes: bytes) -> str:
    if not html_bytes:
        return ""
    bytes_type = detect_encoding(html_bytes)
    if bytes_type is None:
        bytes_type = "latin-1" 
    html_str = html_bytes.decode(bytes_type, errors='ignore')
    text = extract_plain_text(html_str)
    return text.strip()

def identify_language(text: str) -> Tuple[str, float]:
    MODEL_PATH = '/Users/thefoolgy/Desktop/assignment1-basics-main/assignment4-data-main/cs336_data/lid.176.bin'
    _MODEL = fasttext.load_model(MODEL_PATH)
    text = text.replace('\n', ' ').strip()
    if not text:
        return ('unknown', 0)
    labels, probs = _MODEL.predict(text, k=1)
    lang = labels[0].replace('__label__', '')
    conf = float(probs[0])
    return (lang, conf)

def mask_emails(text: str) -> tuple[str, int]:
    EMAIL_PATTERN = re.compile(
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
)
    matches = EMAIL_PATTERN.findall(text)
    count = len(matches)
    masked_text = EMAIL_PATTERN.sub("|||EMAIL_ADDRESS|||", text)
    
    return (masked_text, count)

def mask_phone_numbers(text: str) -> tuple[str, int]:
    PHONE_PATTERN = re.compile(
    r'(?<!\d)(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)'
)
    matches = PHONE_PATTERN.findall(text)
    count = len(matches)
    masked_text = PHONE_PATTERN.sub("|||PHONE_NUMBER|||", text)
    return (masked_text, count)

def mask_ips(text: str) -> tuple[str, int]:
    IPV4_PATTERN = re.compile(
    r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
)
    matches = IPV4_PATTERN.findall(text)
    count = len(matches)
    masked_text = IPV4_PATTERN.sub("|||IP_ADDRESS|||", text)
    return (masked_text, count)

def classify_nsfw(text: str) -> tuple[Any, float]:
    MODEL_PATH = '/Users/thefoolgy/Desktop/assignment1-basics-main/assignment4-data-main/cs336_data/jigsaw_fasttext_bigrams_nsfw_final.bin'
    _MODEL = fasttext.load_model(MODEL_PATH)
    text = text.replace('\n', ' ').strip()
    if not text:
        return ('unknown', 0)
    labels, probs = _MODEL.predict(text, k=1)
    lang = labels[0].replace('__label__', '')
    conf = float(probs[0])
    return (lang, conf)

def classify_toxic_speech(text: str) -> tuple[Any, float]:
    MODEL_PATH = '/Users/thefoolgy/Desktop/assignment1-basics-main/assignment4-data-main/cs336_data/jigsaw_fasttext_bigrams_hatespeech_final.bin'
    _MODEL = fasttext.load_model(MODEL_PATH)
    text = text.replace('\n', ' ').strip()
    if not text:
        return ('unknown', 0)
    labels, probs = _MODEL.predict(text, k=1)
    lang = labels[0].replace('__label__', '')
    conf = float(probs[0])
    return (lang, conf)

def gopher_quality_filter(text: str) -> bool:
    if not text or not text.strip():
        return False
    words = text.split()
    word_count = len(words)
    if word_count < 50 or word_count > 100_000:
        return False
    if words:
        total_length = sum(len(word) for word in words)
        mean_word_length = total_length / len(words)
        if mean_word_length < 3 or mean_word_length > 10:
            return False
    lines = text.split('\n')
    if lines:
        ellipsis_lines = sum(1 for line in lines if line.rstrip().endswith('...'))
        ellipsis_ratio = ellipsis_lines / len(lines)
        if ellipsis_ratio >= 0.30:
            return False  
    if words:
        alphabetic_words = sum(1 for word in words if any(c.isalpha() for c in word))
        alphabetic_ratio = alphabetic_words / len(words)
        if alphabetic_ratio < 0.80:
            return False  
    return True

def classify_quality(text: str) -> tuple[Any, float]:
    MODEL_PATH = '/Users/thefoolgy/Desktop/assignment1-basics-main/assignment4-data-main/cs336_data/quality_classifier.bin'
    _MODEL = fasttext.load_model(MODEL_PATH)
    text = text.replace('\n', ' ').strip()
    labels, probs = _MODEL.predict(text, k=1)
    lang = labels[0].replace('__label__', '')
    conf = float(probs[0])
    return (lang, conf)