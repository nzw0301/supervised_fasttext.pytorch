import sys

str2id = {
    'zero': '0',
    'one': '1',
    'two': '2',
    'three': '3',
    'four': '4'
}

fin_name = sys.argv[1]
num_classes = int(sys.argv[2])
assert num_classes in [2, 5]

replace_patterns = {
    'sentiment-dev.normalized': 'train.tsv',
    'sentiment-test.normalized': 'val.tsv',
    'sentiment-train.normalized': 'test.tsv'
}

fname = fin_name.split('/')[-1]
fout_name = fin_name.replace(fname, replace_patterns[fname])

with open(fin_name) as fin, open(fout_name, 'w') as fout:
    for l in fin:
        l = l.strip()
        words = l.split()

        # binary
        if num_classes == 2:
            label = str2id[words[-1]]
            sentence = ' '.join(words[:-1])
        else: # fine
            label = str2id[words[0]]
            sentence = ' '.join(words[1:])

        l = '{}\t{}\n'.format(label, sentence)
        fout.write(l)
