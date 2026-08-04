import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def clean_text_classic(text, remove_stopwords=False, stemming=False):
    """
    Text preprocessing for classic NLP models.

    Parameters
    ----------
    text : str
        Input text.

    remove_stopwords : bool
        Whether to remove English stopwords.

    stemming : bool
        Whether to apply stemming.

    Returns
    -------
    str
        Cleaned text.
    """

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove mentions
    text = re.sub(r"@\w+", "", text)

    # Keep only letters and spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        text = " ".join(
            word for word in text.split()
            if word not in stop_words
        )

    if stemming:
        stemmer = PorterStemmer()
        text = " ".join(
            stemmer.stem(word)
            for word in text.split()
        )

    return text


def clean_text_transformer(text):
    """
    Minimal preprocessing for Transformer models.

    Stemming, lemmatization, stopword removal
    and punctuation removal are intentionally not used.
    """

    if text is None:
        return ""

    text = str(text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text
