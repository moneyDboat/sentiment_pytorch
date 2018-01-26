import os
from collections import Counter
from past.builtins import xrange
import xml.etree.ElementTree as ET


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
def _get_data_tuple(text, asp_term, fro, to, label, word2idx):
    words = text.split()
    # Find the ids of aspect term
    ids, st, i = [], _count_pre_spaces(text), 0
    for word in words:
        if _check_if_ranges_overlap(st, st + len(word) - 1, fro, to - 1):
            ids.append(i)
        st = st + len(word) + _count_mid_spaces(text, st + len(word))
        i = i + 1
    pos_info, i = [], 0
    for word in words:
        pos_info.append(_get_abs_pos(i, ids))
        i = i + 1
    lab = None
    if label == 'negative':
        lab = 0
    elif label == 'neutral':
        lab = 1
    else:
        lab = 2
    return pos_info, lab


# 读取数据集，target代表aspect
def load_data(fname, source_count, source_word2idx, target_count, target_word2idx):
    if os.path.isfile(fname) is False:
        raise ("[!] Data %s not found" % fname)

    # ElementTree解析XML
    tree = ET.parse(fname)
    root = tree.getroot()

    # context_words, aspect_words
    source_words, target_words, max_sent_len = [], [], 0
    for sentence in root:
        text = sentence.find('text').text.lower()
        source_words.extend(text.split())
        if len(text.split()) > max_sent_len:
            max_sent_len = len(text.split())
        for asp_terms in sentence.iter('aspectTerms'):
            for asp_term in asp_terms.findall('aspectTerm'):
                target_words.append(asp_term.get('term').lower())

    if len(source_count) == 0:
        source_count.append(['<pad>', 0])
    source_count.extend(Counter(source_words).most_common())  # Counter计数，most_common根据计数进行排序
    target_count.extend(Counter(target_words).most_common())

    for word, _ in source_count:
        if word not in source_word2idx:
            source_word2idx[word] = len(source_word2idx)  # 添加单词索引,word2index

    for word, _ in target_count:
        if word not in target_word2idx:
            target_word2idx[word] = len(target_word2idx)

    source_data, source_loc_data, target_data, target_label = list(), list(), list(), list()
    for sentence in root:
        text = sentence.find('text').text.lower()
        if len(text.strip()) != 0:
            idx = []
            for word in text.split():
                idx.append(source_word2idx[word])  # sentence转变成word indexs列表
            for asp_terms in sentence.iter('aspectTerms'):
                for asp_term in asp_terms.findall('aspectTerm'):
                    source_data.append(idx)
                    pos_info, lab = _get_data_tuple(text, asp_term.get('term').lower(), int(asp_term.get('from')),
                                                    int(asp_term.get('to')), asp_term.get('polarity'), source_word2idx)
                    source_loc_data.append(pos_info)
                    target_data.append(target_word2idx[asp_term.get('term').lower()])
                    target_label.append(lab)

    print("Read %s aspects from %s" % (len(source_data), fname))
    return source_data, source_loc_data, target_data, target_label, max_sent_len
