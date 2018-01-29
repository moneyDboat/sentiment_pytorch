import json
import os
import shutil
from collections import Counter
from models.data_model import SentiData
import pickle
import numpy as np

from tqdm import tqdm

emb_size = 300
labels = {'Negative': 0, 'Positive': 1}


def build_vocab(*fpaths):
    source_count, target_count = [], []
    source_word2idx, target_word2idx = {}, {}
    source_words, target_words = [], []
    source_count.append(['<pad>', 0])

    for fpath in fpaths:
        if os.path.isfile(fpath) is False:
            raise ("[!] Data %s not found" % fpath)
        print('<--loading data file {}-->'.format(fpath))

        # 解析json
        with open(fpath, 'r') as f:
            data = json.load(f)
            for item in data:
                text = item['text'].lower()
                source_words.extend(text.strip().split())
                opinions = item['opinions']
                for opinion in opinions:
                    target_words.append(opinion['aspect'].lower())

    source_count.extend(Counter(source_words).most_common())
    target_count.extend(Counter(target_words).most_common())

    # 单词转换成索引，word2index
    # 最常出现的词会在前面
    for word, _ in source_count:
        if word not in source_word2idx:
            source_word2idx[word] = len(source_word2idx)
    for word, _ in target_count:
        if word not in target_word2idx:
            target_word2idx[word] = len(target_word2idx)

    # 写入文件
    with open(file_path + 'source_w2i.txt', 'wt') as f:
        for key, val in source_word2idx.items():
            f.write('\"{}\" : {}\n'.format(key, val))
    with open(file_path + 'target_w2i.txt', 'wt') as f:
        for key, val in target_word2idx.items():
            f.write('\"{}\" : {}\n'.format(key, val))

    return source_word2idx, target_word2idx


def init_word_embeddings(word2idx):
    weight = np.random.normal(0, 0.05, [len(word2idx), emb_size])
    print('<--loading pre-trained word vectors...-->')
    with open('data/glove.6B.300d.txt', 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            content = line.strip().split()
            if content[0] in word2idx:
                weight[word2idx[content[0]]] = np.array(list(map(float, content[1:])))
    return weight


def load_data(fpath, data_type, source_word2idx, target_word2idx):
    if os.path.isfile(fpath) is False:
        raise ("[!] Data %s not found" % fpath)

    raw_sentence = []
    all_data = {}
    source_data, loc_data, target_data, target_label = [], [], [], []

    with open(fpath, 'r') as f:
        data = json.load(f)

        for item in data:
            text = item['text'].strip().lower()
            raw_sentence.append(text)
            sentence_idx = []
            for word in text.split():
                sentence_idx.append(source_word2idx[word])

            for opinion in item['opinions']:
                label = labels[opinion['sentiment']]
                target_label.append(label)
                source_data.append(sentence_idx)
                target_data.append(target_word2idx[opinion['aspect'].lower()])

    # 写入文件
    print("<--Read %s aspects from %s-->" % (len(source_data), fpath))
    source_data = pad_to_batch_max(source_data)
    target_data, target_label = np.array(target_data, dtype=int), np.array(target_label, dtype=int)
    save_path = file_path + data_type
    with open(save_path + '/raw_sentence.txt', 'w') as f:
        for item in raw_sentence:
            f.write(item + '\n')
    np.savetxt(save_path + '/aspects.txt', target_data, fmt='%i')
    np.savetxt(save_path + '/labels.txt', target_label, fmt='%i')

    all_data = SentiData(source_data, loc_data, target_data, target_label)
    return all_data


def pad_to_batch_max(data):
    max_len = max([len(item) for item in data], )
    pad_data = np.zeros([len(data), max_len], dtype=int)

    for i in range(len(data)):
        for j in range(len(data[i])):
            pad_data[i][j] = data[i][j]

    return pad_data


file_path = 'preprocess/sentihood/'
if os.path.isdir(file_path):
    shutil.rmtree(file_path)
os.makedirs(file_path + 'train/')
os.makedirs(file_path + 'test/')

train_data_path = 'data/sentihood-train.json'
dev_data_path = 'data/sentihood-dev.json'
test_data_path = 'data/sentihood-test.json'

source_word2idx, target_word2idx = build_vocab(train_data_path, dev_data_path, test_data_path)
train_data = load_data(train_data_path, 'train', source_word2idx, target_word2idx)
test_data = load_data(test_data_path, 'test', source_word2idx, target_word2idx)

embeddings = init_word_embeddings(source_word2idx)
# 保存预处理数据
preprocess_data = dict()
preprocess_data['embedding'] = embeddings
preprocess_data['train'] = train_data
preprocess_data['test'] = test_data
preprocess_data['source_w2i'] = source_word2idx
preprocess_data['target_w2i'] = target_word2idx
pickle.dump(preprocess_data, open(file_path + 'data.pkl', 'wb'))
