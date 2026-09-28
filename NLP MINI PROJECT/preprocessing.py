"""
preprocessing.py
----------------
Text preprocessing pipeline for Crowd Incident Text Categorization.
Includes:
- Lowercasing
- Noise removal (URLs, mentions, special characters, digits)
- Tokenization
- Stopwords filtering
- Lemmatization (WordNetLemmatizer)
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Ensure essential NLTK resources
try:
    _stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    _stop_words = set(stopwords.words('english'))

try:
    _lemmatizer = WordNetLemmatizer()
    _lemmatizer.lemmatize("testing")
except Exception:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    _lemmatizer = WordNetLemmatizer()

# Preserve critical emergency context words that standard stopwords might drop
CRITICAL_SAFETY_TERMS = {"not", "no", "down", "out", "over", "under", "again", "off"}
STOP_WORDS = _stop_words - CRITICAL_SAFETY_TERMS


def clean_text(text: str) -> str:
    """Removes URLs, mentions, punctuations, and excessive whitespace."""
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # Remove email addresses and user mentions
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)

    # Remove numbers / digits
    text = re.sub(r'\b\d+\b', ' ', text)

    # Remove punctuation
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_text(text: str) -> str:
    """
    Full NLP preprocessing pipeline:
    Cleaning -> Tokenization -> Stopword removal -> Lemmatization -> Joined string
    """
    cleaned = clean_text(text)
    if not cleaned:
        return ""

    # Tokenize
    try:
        tokens = word_tokenize(cleaned)
    except LookupError:
        nltk.download('punkt', quiet=True)
        tokens = word_tokenize(cleaned)
    except Exception:
        # Fallback to whitespace regex tokenization
        tokens = re.findall(r'\b[a-zA-Z]{2,}\b', cleaned)

    # Filter stopwords and short tokens, apply lemmatization
    processed_tokens = []
    for token in tokens:
        if token not in STOP_WORDS and len(token) > 1:
            lemma = _lemmatizer.lemmatize(token)
            processed_tokens.append(lemma)

    return " ".join(processed_tokens)


if __name__ == "__main__":
    sample = "EMERGENCY: Massive crowd surge at gate 4! People are getting trampled & falling down rapidly."
    print("Original:", sample)
    print("Processed:", preprocess_text(sample))
