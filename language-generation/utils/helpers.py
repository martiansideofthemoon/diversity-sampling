"""A set of global utility functions."""
import collections

from strings import WORD_CHAR_DEFAULTS


def generate_vocab(text):
    """
    Generate a vocabulary from `text`.

    Tokens are assumed to be separated by whitespaces. <unk> characters
    have been explicitely removed from this vocabulary generator.
    """
    tokens = text.split()
    counter = collections.Counter(tokens)
    # We do not want to store the <unk>, <s>, </s> character in vocabulary
    del counter["<unk>"]
    del counter["<s>"]
    del counter["</s>"]
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    unique_tokens, _ = zip(*count_pairs)
    vocab = dict(zip(unique_tokens, range(len(unique_tokens))))
    return vocab


def update_vocab(vocab):
    """Add relevant special tokens to vocab."""
    vocab_size = len(vocab)
    vocab["<unk>"] = vocab_size
    vocab["<s>"] = vocab_size + 1
    vocab["</s>"] = vocab_size + 2
    return vocab


def fix_data(text):
    """Add BOS and EOS markers to sentence."""
    if "<s>" in text and "</s>" in text:
        # This hopes that the text has been correct pre-processed
        return text
    sentences = text.split("\n")
    # Removing any blank sentences data
    sentences = ["<s> " + s + " </s>" for s in sentences if len(s.strip()) > 0]
    return " ".join(sentences)


def set_char_defaults(args):
    """Change all the default filenames to char filenames."""
    for w in WORD_CHAR_DEFAULTS:
        if getattr(args, w[0]) == w[1]:
            setattr(args, w[0], w[2])
    return args


class Node(object):
    """A single node of trie object."""

    def __init__(self, key, depth=0):
        """Build a Node object."""
        self.parent = None
        # `children` is a letter to Node mapping
        self.children = {}
        self.key = key
        self.frequency = {}
        for i in range(depth + 1):
            self.frequency[i] = 0
