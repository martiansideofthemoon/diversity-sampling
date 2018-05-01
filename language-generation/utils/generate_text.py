import argparse
import numpy as np
import os
import sys

from decoder import trie


def weighted_pick(weights):
    t = np.cumsum(weights)
    s = np.sum(weights)
    return(int(np.searchsorted(t, np.random.rand(1) * s)))

parser = argparse.ArgumentParser()
parser.add_argument('--vocab', type=str, default="vocab", help='Use SRILM processed vocabulary')
parser.add_argument('--dataset', type=str, default="ptb", help='Dataset for LM experiments')
parser.add_argument('--lm', type=str, default="LM", help='Use SRILM processed vocabulary')
parser.add_argument('--tokens', type=int, default=100, help='Number of tokens to be generated')
parser.add_argument('--input', type=str, default="my", help='Number of tokens to be generated')
parser.add_argument('--outfile', type=str, default="generated.txt", help='Number of tokens to be generated')
parser.add_argument('--data_dir', type=str, default='data', help='data directory containing input.txt')
args = parser.parse_args()

lm_path = os.path.join(args.data_dir, args.lm)
saved_vocab = os.path.join(args.data_dir, args.vocab)
with open(saved_vocab, 'r') as f:
    rev_vocab = f.read().split()

vocab = {word: i for i, word in enumerate(rev_vocab)}
vocab_size = len(vocab)

# Load n-gram ARPA file
tr = trie.Trie()
tr.load_arpa(lm_path, trie.map_string_int(vocab))

context = [vocab[x] if x in vocab else vocab['<unk>'] for x in args.input.split()]

output = ' '.join([rev_vocab[x] for x in context])

if len(context) < 3:
    context = [vocab['<s>']] + context
else:
    context = context[-2:]

for i in range(args.tokens):
    if i % 100 == 0:
        print("%d / %d tokens generated" % (i, args.tokens))
    context_wrapped = trie.list_int(context)
    tensor = np.zeros(vocab_size)
    tr.get_distro(context_wrapped, tensor)
    token = weighted_pick(np.exp(tensor))
    if token == vocab_size:
        token = token - 1
    if token == vocab['</s>']:
        # Save output string into a file
        context = [vocab['<s>']]
        with open(args.outfile, 'a') as f:
            f.write(output.strip() + "\n")
        output = ''
    elif token == vocab['<s>']:
        print("Error!")
        sys.exit(0)
    else:
        output += ' ' + rev_vocab[token]
        if len(context) == 1:
            context = context + [token]
        elif len(context) == 2:
            context = context[1:] + [token]
        else:
            print("Error 2")
            sys.exit(0)
