from sapir_image_captions import UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN


def text2tensor(text, vocab, max_seq_length=50):
    """Convert string text into numerical tensor.

    Parameters
    ----------
    text: list[str]
        List of tuples of strs
    vocab: torchtext.data.Vocab object
        Vocab contains itos and stoi
    max_seq_length: int [Default: 50]
        Max number of tokens.

    """
    UNK_IDX = vocab.stoi[UNK_TOKEN]
    SOS_IDX = vocab.stoi[SOS_TOKEN]
    EOS_IDX = vocab.stoi[EOS_TOKEN]
    PAD_IDX = vocab.stoi[PAD_TOKEN]

    text_ = [SOS_IDX] + [vocab.stoi.get(ch, UNK_IDX) for ch in text]
    text_ = text_[:max_seq_length - 1] + [EOS_IDX]
    text_ = text_ + [PAD_IDX] * (max_seq_length - len(text_))
    return text_
