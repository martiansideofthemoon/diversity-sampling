"""This code generates n-gram files."""
import re
import os
import subprocess
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default='input.txt',
                    help='Filename having textual data in data_dir.')
parser.add_argument('--outfile', type=str, default='output.txt',
                    help='Filename having textual data in data_dir.')
parser.add_argument('--lm', type=str, default='LM',
                    help='Filename having textual data in data_dir.')
args = parser.parse_args()

with open(args.filename, 'r') as f:
    data = f.readlines()
for i, line in enumerate(data):
    data[i] = data[i] + " </s>"
data = "\n".join(data).split()

# This RE is used to parse output produced by SRILM
regex = re.compile(r'\sp\(\s(.*)\s\|.*\]\s(.*)\s\[')
srilm = '/share/data/speech/Software/SRILM/bin/i686-m64'
ngram = os.path.join(srilm, 'ngram')

lm_file = args.lm
command = \
    ngram + " " + \
    "-unk " + \
    "-order " + str(3) + " " + \
    "-lm " + lm_file + " " + \
    "-debug 2 " + \
    "-ppl " + args.filename

results = subprocess.check_output(command,
                                  stderr=subprocess.STDOUT,
                                  shell=True)
results = results.split('\n')
token_ptr = 0
output = ""
for result in results:
    match = regex.search(result)
    if not match:
        continue
    if data[token_ptr] == '<s>':
        token_ptr += 1
        output += "<s> 1.0\n"
    if token_ptr == 0:
        # Ignoring the first word
        token_ptr += 1
        continue
    active_token = data[token_ptr]
    # Confirm active_token and matched token are same!
    if active_token != match.group(1) and match.group(1) != '<unk>':
        print "Error! " + active_token + " " + match.group(1)
        sys.exit()
    output += active_token + " " + match.group(2) + "\n"
    token_ptr += 1
with open(args.outfile, 'w') as f:
    f.write(output)
