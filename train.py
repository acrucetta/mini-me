import utils.preprocessing as preprocessing
from models.models import Encoder, Decoder, Seq2Seq

BASE_FOLDER = "/Users/andrescrucettanieto/Library/CloudStorage/OneDrive-WaltzHealth/andrescrucettanieto/andres-vault/Areas/journal/diary/"


def train():
    data_loader, vocab = preprocessing.prepare_data(BASE_FOLDER)
    input_size = output_size = len(vocab)
    hidden_size = 256
    encoder = Encoder(input_size, hidden_size)
    decoder = Decoder(hidden_size, output_size)
    model = Seq2Seq(encoder, decoder).to(device)
