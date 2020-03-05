import gzip
import os
import numpy as np
import pickle
import time


def convert_emb(language, ft_dir):
    fasttext_embeddings_index = {}

    if language == 'ZH':
        ft_file = os.path.join(ft_dir, 'cc.zh.300.vec.gz')
        if os.path.isfile(ft_file):
            f = gzip.open(ft_file, 'rb')

    if language == 'EN':
        ft_file = os.path.join(ft_dir, 'cc.en.300.vec.gz')
        if os.path.isfile(ft_file):
            f = gzip.open(ft_file, 'rb')

    if f is None:
        raise ValueError("Embedding not found")

    for line in f.readlines():
        values = line.strip().split()
        word = values[0].decode('utf8')
        coefs = np.asarray(values[1:], dtype='float32')
        fasttext_embeddings_index[word] = coefs

    return fasttext_embeddings_index


def save_emb(language, ft_dir, emb):
    file_path = os.path.join(ft_dir, language+'.pkl')
    pickle.dump(emb, open(file_path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)


def load_emb(language, ft_dir):
    file_path = os.path.join(ft_dir, language+'.pkl')
    emb = pickle.load(open(file_path, "rb"))
    return emb


if __name__ == "__main__":
    language = 'ZH'
    ft_dir = '/home/dingsda/data/embedding'

    print('convert')
    t1 = time.time()
    emb1 = convert_emb(language, ft_dir)
    t2 = time.time()
    print(t2-t1)

    print('save')
    save_emb(language, ft_dir, emb1)
    t3 = time.time()
    print(t3-t2)

    print('load')
    emb2 = load_emb(language, ft_dir)
    t4 = time.time()
    print(t4-t3)

    print(len(emb1))
    print(len(emb2))





