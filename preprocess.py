import os
from collections import Counter
from past.builtins import xrange
import xml.etree.ElementTree as ET
import argparse
import json
import numpy as np
import pickle
from models.data_model import SentiData
import util
import shutil

# 参数配置
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', dest='dataset', type=str, metavar='<str>', default='Restaurants',
                    help='Dataset (Laptop/Restaurants) (default=Restaurants)')
parser.add_argument('--glove', dest='glove', type=str, default='data/glove.6B.300d.txt',
                    help='glove path')
args = parser.parse_args()

labels = {'negative': 0, 'neutral': 1, 'positive': 2}

file_path = 'preprocess/{}/'.format(args.dataset)
if os.path.isdir(file_path):
    shutil.rmtree(file_path)
os.makedirs(file_path + 'train/')
os.makedirs(file_path + 'test/')


def main():
    train_data_path = 'data/{}_Train.xml'.format(args.dataset)
    test_data_path = 'data/{}_Test.xml'.format(args.dataset)

    source_word2idx, target_word2idx = build_voca(train_data_path, test_data_path)
    train_data = load_data(train_data_path, 'train', source_word2idx, target_word2idx)
    test_data = load_data(test_data_path, 'test', source_word2idx, target_word2idx)

    # 保存预处理数据
    preprocess_data = dict()
    preprocess_data['train'] = train_data
    preprocess_data['test'] = test_data
    preprocess_data['source_w2i'] = source_word2idx
    preprocess_data['target_w2i'] = target_word2idx
    pickle.dump(preprocess_data, open(file_path + 'data.pkl', 'wb'))


# 构建词典
def build_voca(*fpaths):
    source_count, target_count = [], []
    source_word2idx, target_word2idx = {}, {}
    source_words, target_words = [], []
    source_count.append(['<pad>', 0])

    for fpath in fpaths:
        if os.path.isfile(fpath) is False:
            raise ("[!] Data %s not found" % fpath)
        print('<--loading data file {}-->'.format(fpath))

        # ElementTree解析XML
        tree = ET.parse(fpath)
        root = tree.getroot()

        # context_words, aspect_words
        for sentence in root:
            text = sentence.find('text').text.lower()
            source_words.extend(text.split())
            for asp_terms in sentence.iter('aspectTerms'):
                for asp_term in asp_terms.findall('aspectTerm'):
                    target_words.append(asp_term.get('term').lower())

    # Counter计数，most_common根据计数进行排序
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


# 读取数据集，target代表aspect
def load_data(fpath, data_type, source_word2idx, target_word2idx):
    if os.path.isfile(fpath) is False:
        raise ("[!] Data %s not found" % fpath)

    # ElementTree解析XML
    tree = ET.parse(fpath)
    root = tree.getroot()

    raw_sentence = []
    all_data = {}
    source_data, loc_data, target_data, target_label = [], [], [], []
    for sentence in root:
        text = sentence.find('text').text.lower()
        if len(text.strip()) != 0:
            raw_sentence.append(text)
            sentence_idx = []
            for word in text.split():
                sentence_idx.append(source_word2idx[word])  # sentence转变成word indexs列表

            for asp_terms in sentence.iter('aspectTerms'):
                for asp_term in asp_terms.findall('aspectTerm'):
                    labe = asp_term.get('polarity')
                    if labe == 'conflict':
                        continue
                    source_data.append(sentence_idx)
                    # 提取postion information和aspect label
                    pos_info, label = _get_data_tuple(text, int(asp_term.get('from')),
                                                      int(asp_term.get('to')), labe)
                    loc_data.append(pos_info)
                    target_data.append(target_word2idx[asp_term.get('term').lower()])
                    target_label.append(label)

    # 写入文件
    print("<--Read %s aspects from %s-->" % (len(source_data), fpath))
    source_data = util.pad_to_batch_max(source_data)
    save_path = file_path + data_type
    with open(save_path + '/raw_sentence.txt', 'w') as f:
        for item in raw_sentence:
            f.write(item + '\n')
    np.savetxt(save_path + '/aspects.txt', np.array(target_data), fmt='%i')
    np.savetxt(save_path + '/labels.txt', np.array(target_label), fmt='%i')

    all_data = SentiData(source_data, loc_data, target_data, target_label)
    return all_data


def _get_abs_pos(cur, ids):
    min_dist = 1000
    for i in ids:
        if abs(cur - i) < min_dist:
            min_dist = abs(cur - i)
    if min_dist == 1000:
        raise ("[!] ids list is empty")
    return min_dist


# 检测text中空格数
def _count_pre_spaces(text):
    count = 0
    for i in xrange(len(text)):
        if text[i].isspace():
            count = count + 1
        else:
            break
    return count


def _count_mid_spaces(text, pos):
    count = 0
    for i in xrange(len(text) - pos):
        if text[pos + i].isspace():
            count = count + 1
        else:
            break
    return count


def _check_if_ranges_overlap(x1, x2, y1, y2):
    return x1 <= y2 and y1 <= x2


# 提取postion information和aspect label
# 这部分代码没有仔细看，但已经知道输出是什么样的了
# pos_info形式类似[5,4,3,2,1,0,1,2,3,4,5]
def _get_data_tuple(text, fro, to, label):
    words = text.split()
    # Find the ids of aspect term
    ids, st, i = [], _count_pre_spaces(text), 0
    for word in words:
        if _check_if_ranges_overlap(st, st + len(word) - 1, fro, to - 1):
            ids.append(i)
        st = st + len(word) + _count_mid_spaces(text, st + len(word))
        i = i + 1
    pos_info, i = [], 0
    for _ in words:
        pos_info.append(_get_abs_pos(i, ids))
        i = i + 1

    return pos_info, labels[label]


if __name__ == '__main__':
    main()
