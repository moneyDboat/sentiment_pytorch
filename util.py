# -*- coding: UTF-8 -*-

import numpy as np
from models.data_model import SentiData
import torch
from torch.autograd import Variable



def make_batch(data, i, batch_size, cuda):
    # -1 to take all
    if (i >= 0):
        start, end = i * batch_size, (i + 1) * batch_size
        batch = SentiData(data.source[start:end, :], data.location[start:end], data.target[start:end],
                          data.label[start:end])
    else:
        batch = data

    sentences = torch.LongTensor(batch.source).transpose(0, 1)
    labels = torch.LongTensor(batch.label.tolist())

    actual_batch = sentences.size(1)
    if (cuda):
        sentences = sentences.cuda()
        labels = labels.cuda()
    sentences = Variable(sentences)
    labels = Variable(labels)

    return sentences, labels, actual_batch
