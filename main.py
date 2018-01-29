import pickle
import argparse
from collections import Counter

from sklearn.metrics import accuracy_score

from models.rnn import BasicRNN
import torch
from torch.autograd import Variable
from torch import optim
from torch import nn
import time
import numpy as np
from models.data_model import SentiData

# 参数配置
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', dest='dataset', type=str, metavar='<str>', default='Restaurants',
                    help="Dataset (Laptop/Restaurants) (default=Restaurants)")
parser.add_argument("--mode", dest="mode", type=str, metavar='<str>', default='term',
                    help="Experiment Mode (term|source) (default=term)")
parser.add_argument("--mdl", dest="model_type", type=str, metavar='<str>', default='RNN',
                    help="(RNN|TD-RNN|ATT-RNN)")
parser.add_argument("--opt", dest="opt", type=str, metavar='<str>', default='Adam',
                    help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=rmsprop)")
parser.add_argument("--lr", dest='learn_rate', type=float, metavar='<float>', default=0.001,
                    help="Learning Rate")
parser.add_argument("--rnn_type", dest="rnn_type", type=str, metavar='<str>', default='RNN',
                    help="Recurrent unit type (RNN|LSTM|GRU) (default=RNN)")
parser.add_argument("--rnn_size", dest="rnn_size", type=int, metavar='<int>', default=300,
                    help="RNN dimension. '0' means no RNN layer (default=300)")
parser.add_argument("--rnn_layers", dest="rnn_layers", type=int, metavar='<int>', default=1,
                    help="Number of RNN layers")
parser.add_argument("--emb_size", dest="emb_size", type=int, metavar='<int>', default=300,
                    help="Embeddings dimension (default=300)")
parser.add_argument("--batch_size", dest="batch_size", type=int, metavar='<int>', default=256,
                    help="Batch size (default=256)")
parser.add_argument("--pretrained", dest="pretrained", type=int, metavar='<int>', default=1,
                    help="Whether to use pretrained or not")

parser.add_argument('--cuda', action='store_true', default=True, help='use CUDA')
parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=100,
                    help="Number of epochs (default=50)")
parser.add_argument('--gpu', dest='gpu', type=int, metavar='<int>', default=0,
                    help="Specify which GPU to use (default=0)")
parser.add_argument('--seed', type=int, default=1111, help='random seed')
args = parser.parse_args()


def train(data):
    num_batches = int(len(data) / args.batch_size) + 1
    select_optimizer()

    print("<---Starting training--->")
    for epoch in range(1, args.epochs + 1):
        t0 = time.clock()
        # random.shuffle(self.train_data)
        print("========================================================================")
        losses = []

        for i in range(num_batches):
            loss = 0
            loss = train_batch(data, i)
            if (loss is None):
                continue
            losses.append(loss)

        t1 = time.clock()
        print("[Epoch {}] Train Loss={} T={}s".format(epoch, np.mean(losses), t1 - t0))


def train_batch(data, i):
    # Trains a regular RNN model
    sentences, sources, actual_batch = make_batch(data, i, args.batch_size, args.cuda)

    hidden = model.init_hidden(actual_batch)
    hidden = model.repackage_hidden(hidden)
    model.zero_grad()
    output, hidden = model.forward(sentences, hidden)
    loss = criterion(output, sources)
    loss.backward()
    optimizer.step()
    return loss.data[0]


def make_batch(data, i, batch_size, cuda):
    # -1 to take all
    if (i >= 0):
        start, end = i * batch_size, (i + 1) * batch_size
        batch = data[start:end]
    else:
        batch = data

    sentences = torch.LongTensor(pad_to_batch_max([item.source for item in batch])).transpose(0, 1)
    labels = torch.LongTensor(np.array([item.label for item in batch]))

    actual_batch = sentences.size(1)
    if (cuda):
        sentences = sentences.cuda()
        labels = labels.cuda()
    sentences = Variable(sentences)
    labels = Variable(labels)

    return sentences, labels, actual_batch


def pad_to_batch_max(text_data):
    max_len = max([len(item) for item in text_data])
    pad_data = np.zeros([len(text_data), max_len], dtype=int)

    for i in range(len(text_data)):
        for j in range(len(text_data[i])):
            pad_data[i][j] = text_data[i][j]

    return pad_data


def evaluate(data):
    # Evaluates normal RNN model
    hidden = model.init_hidden(len(test_data))
    sentence, sources, actual_batch = make_batch(test_data, -1, args.batch_size, args.cuda)
    output, hidden = model.forward(sentence, hidden)
    loss = criterion(output, sources)
    print("Test loss={}".format(loss[0]))
    accuracy = get_accuracy(output, sources)


def get_accuracy(output, sources):
    output = output.data.cpu().numpy()  # （1120,3）
    sources = sources.data.cpu().numpy()
    output = np.argmax(output, axis=1)  # (1120,1)
    dist = dict(Counter(output))
    print("Output Distribution={}".format(dist))
    acc = accuracy_score(sources, output)
    print("Accuracy={}".format(acc))
    return acc


def select_optimizer():
    if (args.opt == 'Adam'):
        op = optim.Adam(model.parameters(), lr=args.learn_rate)
    elif (args.opt == 'RMS'):
        op = optim.RMSprop(model.parameters(), lr=args.learn_rate)
    elif (args.opt == 'SGD'):
        op = optim.SGD(model.parameters(), lr=args.learn_rate)
    elif (args.opt == 'Adagrad'):
        op = optim.Adagrad(model.parameters(), lr=args.learn_rate)
    elif (args.opt == 'Adadelta'):
        op = optim.Adadelta(model.parameters(), lr=args.learn_rate)

    return op


args.dataset = 'sentihood'
data = pickle.load(open('preprocess/{}/data.pkl'.format(args.dataset), 'rb'))
source_w2i, source_w2i, train_data, test_data, glove_weight = data['source_w2i'], data['source_w2i'], \
                                                              data['train'], data['test'], data['embedding']

model = BasicRNN(args, len(source_w2i), pretrained=glove_weight)
print(model)
if args.cuda:
    model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = select_optimizer()

train(train_data)
evaluate(test_data)
