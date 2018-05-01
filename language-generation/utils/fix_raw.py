"""This script fixes the raw data mined from wikipedia."""
import argparse
import codecs
import collections
import os


def process_char(data):
    """Convert a word string into corresponding char string."""
    data = data.replace(" ", "_")
    sentences = data.split("\n")
    output = ""
    spc = []
    for i, x in enumerate(sentences):
        spc.append(" ".join(list(x)))
    output = "\n".join(spc)
    return output

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default="data",
                    help='The location of all the data')
parser.add_argument('--input_file', type=str, default='input.txt',
                    help='Input file to be pruned')
parser.add_argument('--word_output', type=str, default='output.txt',
                    help='Word output file')
parser.add_argument('--word_output_unk', type=str, default='output_unk.txt',
                    help='Word output file')
parser.add_argument('--char_output', type=str, default='char_output.txt',
                    help='Character output file')
parser.add_argument('--char_output_unk', type=str, default='char_output_unk.txt',
                    help='Character output file')
parser.add_argument('--lang', type=str, default='ml',
                    help='Character output file')
parser.add_argument('--unk_limit', type=int, default=1,
                    help='Used to limit the number of <unk> words')
args = parser.parse_args()

input_path = os.path.join(args.data_dir, args.input_file)
word_output = os.path.join(args.data_dir, args.word_output)
word_output_unk = os.path.join(args.data_dir, args.word_output_unk)
char_output = os.path.join(args.data_dir, args.char_output)
char_output_unk = os.path.join(args.data_dir, args.char_output_unk)

with codecs.open(input_path, "r", "utf-8") as f:
    data = f.read()

if args.lang == "ml":
    args.separator = "."
elif args.lang == "hi":
    args.separator = u'\u0964'
else:
    args.separator = "."

sentences = data.split(args.separator)
# Removing the very short sentences
sentences = [" ".join(x.split()) for x in sentences if len(x.split()) > 2]
output = "\n".join(sentences)
char_out = process_char(output)

tokens = output.split()
counter = collections.Counter(tokens)
unk_words = {}
for word in counter:
    if counter[word] <= args.unk_limit:
        unk_words[word] = 1

spc = []
for s in sentences:
    words = s.split()
    for i, w in enumerate(words):
        if w in unk_words:
            words[i] = "<unk>"
    spc.append(" ".join(words))
output_unk = "\n".join(spc)
char_unk = process_char(output_unk)

with codecs.open(word_output, 'w', 'utf-8') as f:
    f.write(output)
with codecs.open(word_output_unk, 'w', 'utf-8') as f:
    f.write(output_unk)
with codecs.open(char_output, 'w', 'utf-8') as f:
    f.write(char_out)
with codecs.open(char_output_unk, 'w', 'utf-8') as f:
    f.write(char_unk)
