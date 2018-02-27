import json
import os
import shutil
from collections import Counter
from models.data_model import SentiData
import pickle
import numpy as np
from tqdm import tqdm
import re

emb_size = 200
labels = {'Negative': 0, 'Positive': 1}


def build_vocab(*fpaths):
    source_count, aspect_count = [], []
    source_w2i, aspect_w2i = {}, {}
    source_words, aspect_words = [], []
    source_count.append(['<pad>', 0])

    for fpath in fpaths:
        if os.path.isfile(fpath) is False:
            raise ("[!] Data %s not found" % fpath)
        print('<--loading data file {}-->'.format(fpath))

        # 解析json
        with open(fpath, 'r') as f:
            data = json.load(f)
            for item in data:
                text = clean_text(item['text'])
                source_words.extend(text.split())
                opinions = item['opinions']
                for opinion in opinions:
                    aspect_words.append(opinion['aspect'].lower())

    source_count.extend(Counter(source_words).most_common())
    aspect_count.extend(Counter(aspect_words).most_common())

    # 单词转换成索引，word2index
    # 最常出现的词会在前面
    for word, _ in source_count:
        if word not in source_w2i:
            source_w2i[word] = len(source_w2i)
    for word, _ in aspect_count:
        if word not in aspect_w2i:
            aspect_w2i[word] = len(aspect_w2i)

    # 写入文件
    with open(file_path + 'source_w2i.txt', 'wt') as f:
        for key, val in source_w2i.items():
            f.write('\"{}\" : {}\n'.format(key, val))
    with open(file_path + 'aspect_w2i.txt', 'wt') as f:
        for key, val in aspect_w2i.items():
            f.write('\"{}\" : {}\n'.format(key, val))

    return source_w2i, aspect_w2i


def clean_text(text):
    text = text.strip().lower()
    # text = re.sub('[-,.():/`;0123456789]+', '', text)
    text = re.sub('[^a-z]+', ' ', text)

    return text


def init_word_embeddings(word2idx):
    weight = np.random.normal(0, 0.05, [len(word2idx), emb_size])
    print('<--loading pre-trained word vectors...-->')
    with open('data/glove/glove.twitter.27B.{}d.txt'.format(emb_size), 'r') as f:
        word2idx_miss = word2idx.copy()
        lines = f.readlines()
        for line in tqdm(lines):
            content = line.strip().split()
            if content[0] in word2idx:
                word2idx_miss.pop(content[0])
                weight[word2idx[content[0]]] = np.array(list(map(float, content[1:])))

    print('<--Missing {}/{} words!-->'.format(len(word2idx_miss), len(word2idx)))
    with open(file_path + 'missing_words.txt', 'w') as f:
        for k, v in word2idx_miss.items():
            f.write("{} : {}\n".format(k, v))
    return weight


def load_data(fpath, data_type, source_w2i, aspect_w2i):
    if os.path.isfile(fpath) is False:
        raise ("[!] Data %s not found" % fpath)

    raw_sentence = []
    source_data, loc_data, aspect_data, aspect_label = [], [], [], []

    with open(fpath, 'r') as f:
        data = json.load(f)

        for item in data:
            text = clean_text(item['text'])
            raw_sentence.append(text)
            sentence_idx = []
            for word in text.split():
                sentence_idx.append(source_w2i[word])

            for opinion in item['opinions']:
                label = labels[opinion['sentiment']]
                aspect_label.append(label)
                source_data.append(sentence_idx)
                aspect_data.append(aspect_w2i[opinion['aspect'].lower()])

    # 写入文件
    print("<--Read %s aspects from %s-->" % (len(source_data), fpath))
    aspect_data, aspect_label = np.array(aspect_data, dtype=int), np.array(aspect_label, dtype=int)
    save_path = file_path + data_type
    with open(save_path + '/raw_sentence.txt', 'w') as f:
        for item in raw_sentence:
            f.write(item + '\n')
    np.savetxt(save_path + '/aspects.txt', aspect_data, fmt='%i')
    np.savetxt(save_path + '/labels.txt', aspect_label, fmt='%i')

    all_data = []
    for i in range(len(source_data)):
        all_data.append(SentiData(source_data[i], None, aspect_data[i], aspect_label[i]))

    all_data = all_data
    return all_data


def get_pos_info():
    pass


file_path = 'preprocess/sentihood/'
if os.path.isdir(file_path):
    shutil.rmtree(file_path)
os.makedirs(file_path + 'train/')
os.makedirs(file_path + 'test/')
os.makedirs(file_path + 'dev/')

train_data_path = 'data/sentihood-train.json'
dev_data_path = 'data/sentihood-dev.json'
test_data_path = 'data/sentihood-test.json'

source_word2idx, aspect_word2idx = build_vocab(train_data_path, dev_data_path, test_data_path)
train_data = load_data(train_data_path, 'train', source_word2idx, aspect_word2idx)
dev_data = load_data(dev_data_path, 'dev', source_word2idx, aspect_word2idx)
test_data = load_data(test_data_path, 'test', source_word2idx, aspect_word2idx)

embeddings = init_word_embeddings(source_word2idx)
# 保存预处理数据
preprocess_data = dict()
preprocess_data['embedding'] = embeddings
preprocess_data['train'] = train_data
preprocess_data['dev'] = dev_data
preprocess_data['test'] = test_data
preprocess_data['source_w2i'] = source_word2idx
preprocess_data['aspect_w2i'] = aspect_word2idx
pickle.dump(preprocess_data, open(file_path + 'data.pkl', 'wb'))
