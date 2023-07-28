import os
from string import punctuation
from typing import List
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langdetect import detect
from googletrans import Translator


BASE_FOLDER = "/Users/andrescrucettanieto/Library/CloudStorage/OneDrive-WaltzHealth/andrescrucettanieto/andres-vault/Areas/journal/"


def get_journal_entries(base_folder: str) -> List:
    """
    This function will load the .md and .txt files
    from my journal folder.
    """
    journal_entries = []
    for filename in os.listdir(base_folder):
        if filename.endswith(".md") or filename.endswith(".txt"):
            with open(base_folder + filename, "r", encoding="utf-8") as f:
                text = f.read()
                journal_entries.append(clean_text(text))
    return journal_entries


def clean_text(text: str) -> str:
    """
    This function will clean the journal text

    We will:
    - Lowercase
    - Remove punctuation
    - Check which language it is
        - If spanish translate to english
        - If english do nothing
    - Tokenize
    - Remove stopwords
    """
    text = text.lower()
    text = "".join([c for c in text if c not in punctuation])

    try:
        language = detect(text)
    except Exception:
        print("Error detecting language, defaulting to english")
        language = "en"

    if language != "en":
        translator = Translator()
        text = translator.translate(text, dest="en").text
    words = word_tokenize(text)
    return words
