from collections import Counter
from itertools import chain
import os
from string import punctuation
from typing import List
import nltk
from nltk.tokenize import word_tokenize
from langdetect import detect
from googletrans import Translator
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]
nltk.download("punkt")


class Vocabulary:
    def __init__(self, special_tokens=None):
        self.w2i = {}
        self.i2w = {}
        self.special_tokens = special_tokens if special_tokens else []
        self.start_index = len(self.special_tokens)

    def build_vocabulary(self, sentences: List[str]):
        words = list(chain(*sentences)) + self.special_tokens
        word_freq = Counter(words)
        self.w2i = {
            word: i + self.start_index
            for i, (word, _) in enumerate(word_freq.most_common())
        }
        self.i2w = {i: word for word, i in self.w2i.items()}

    def __len__(self):
        return len(self.w2i)


def get_journal_entries(base_folder: str) -> List[str]:
    """
    This function will load the .md and .txt files
    from my journal folder.
    """
    journal_entries = []
    for filename in os.listdir(base_folder):
        # If it's a directory, we search inside it
        if os.path.isdir(base_folder + filename):
            journal_entries += get_journal_entries(base_folder + filename + "/")
        # If it's a file, we read it
        if filename.endswith(".md") or filename.endswith(".txt"):
            with open(base_folder + filename, "r", encoding="utf-8") as f:
                print(f"Reading file {filename}")
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
        translation = translator.translate(text, src="es", dest="en")
        text = translation.text

    sentence = "<SOS> " + " ".join(word_tokenize(text)) + " <EOS>"
    return sentence


def build_vocabulary(sentences: List[str]) -> Vocabulary:
    vocab = Vocabulary(SPECIAL_TOKENS)
    vocab.build_vocabulary(sentences)
    return vocab


def text_to_sequence(text: str, vocab: Vocabulary) -> List[int]:
    return [vocab.w2i.get(word, vocab.w2i["<UNK>"]) for word in text.split()]


class TextDataset(Dataset):
    def __init__(self, sentences: List[List[str]]):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx: int):
        return self.sentences[idx]


def prepare_data(base_folder: str):
    sentences = get_journal_entries(base_folder)
    vocab = build_vocabulary(sentences)
    sequences = [text_to_sequence(text, vocab) for text in sentences]

    # Pad sequences and create PyTorch DataLoader
    sequences_padded = pad_sequence(
        [torch.tensor(seq) for seq in sequences],
        batch_first=True,
        padding_value=vocab.w2i["<PAD>"],
    )
    dataset = TextDataset(sequences_padded)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    return data_loader, vocab


if __name__ == "__main__":
    data_loader, vocab = prepare_data(
        base_folder="/Users/andrescrucettanieto/Library/CloudStorage/OneDrive-WaltzHealth/andrescrucettanieto/andres-vault/Areas/journal/diary/"
    )
    for batch in data_loader:
        print(batch)
        break
