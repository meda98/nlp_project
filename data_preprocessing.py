import re
import os
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# Get the path of the directory where the project is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Tell NLTK to look for its data files in the subfolder "nltk_data"
nltk.data.path.append(os.path.join(BASE_DIR, "nltk_data"))

# Initialize the standard NLTK stopword list for English
STOP_WORDS = set(stopwords.words("english"))

# Keep negation-related words
STOP_WORDS.discard("not")
STOP_WORDS.discard("no")
STOP_WORDS.discard("nor")

# Initialize WordNet lemmatizer for converting words to base forms
LEMMATIZER = WordNetLemmatizer()

def get_wordnet_pos(tag: str):
    """
    Converts an NLTK POS tag to the corresponding WordNet POS tag.

    Args:
        tag: Part-of-speech tag produced by NLTK POS tagging.

    Returns:
        Matching WordNet POS tag used for lemmatization.
    """
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
CONTRACTIONS = {
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "can't": "can not",
    "couldn't": "could not",
    "won't": "will not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "i've": "i have",
    "i'm": "i am",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "ain't": "is not"
}

def expand_contractions(text: str) -> str:
    """
    Replaces English contractions with their expanded forms.

    Args:
        text: Raw text containing contractions.

    Returns:
        Text with contractions expanded.
    """

    for contraction, expanded in CONTRACTIONS.items():
        text = re.sub(rf"\b{re.escape(contraction)}\b", expanded, text)

    return text

def clean_text(text: str) -> str:
    """
    Applies preprocessing to a single movie review by converting text to
    lowercase, expanding contractions, removing HTML line break tags,
    preserving hyphenated compounds with underscores, removing unwanted
    characters, normalizing whitespace, tokenizing the text, removing
    stopwords, and performing POS-aware lemmatization.

    Args:
        text: Raw review text.

    Returns:
        Cleaned review text.
    """

    # convert text to lowercase
    text = text.lower()

    # expand contractions
    text = expand_contractions(text)

    # remove HTML line break tags
    text = re.sub(r"<br\s*/?>", " ", text)

    # preserve hyphenated compounds
    text = re.sub(r"(\w)-(\w)", r"\1_\2", text)

    # remove remaining characters except letters, underscores, and whitespace
    text = re.sub(r"[^a-z_\s]", " ", text)

    # normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

   # tokenize text
    tokens = text.split()

    # remove stopwords
    tokens = [w for w in tokens if w not in STOP_WORDS]

   # POS-aware lemmatization
    pos_tags = pos_tag(tokens)

    tokens = [
        LEMMATIZER.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in pos_tags
    ]

    # join words back into a string
    text = " ".join(tokens)

    return text


def preprocess_reviews(reviews: list[str]) -> list[str]:
    """
    Applies preprocessing to a list of reviews.

    Args:
        reviews: List of raw review texts.

    Returns:
        List of cleaned review texts.
    """

    return [clean_text(review) for review in reviews]