import numpy as np


def pad_to_batch_max(data):
    max_len = max([len(item) for item in data], )
    pad_data = np.zeros([len(data), max_len], dtype=int)

    for i in range(len(data)):
        for j in range(len(data[i])):
            pad_data[i][j] = data[i][j]

    return pad_data


def make_batch(data, i, batch_size):
    # -1 to take all
    if (i >= 0):
        batch = data[i * batch_size: (i + 1) * batch_size]
    else:
        batch = data

    sentence = self.pad_to_batch_max([data['tokenized_txt'] for data in batch])  # 填充序列
    targets = torch.LongTensor(np.array([data['polarity'] for data in batch], dtype=np.int32).tolist())
    actual_batch = sentence.size(1)
    if (self.args.cuda):
        sentence = sentence.cuda()
        targets = targets.cuda()
    sentence = Variable(sentence)
    targets = Variable(targets)
    return sentence, targets, actual_batch
