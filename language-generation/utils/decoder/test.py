import numpy as np
import time
import trie

with open('data/vocab', 'r') as f:
    rev_vocab = f.read().split()

vocab = {v: i for i, v in enumerate(rev_vocab)}
vocab = trie.map_string_int(vocab)

tr = trie.Trie()
tr.load_arpa('data/LM', vocab)
word = trie.list_int([0, 238])

start = time.time()
distro = np.zeros((32, 35, len(vocab)))
for i in range(32):
    for j in range(35):
        tr.get_distro(word, distro[i][j])

print(time.time() - start)
