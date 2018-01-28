import pickle
import argparse
from models.rnn import BasicRNN
import numpy as np
from tqdm import tqdm

# 参数配置
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', dest='dataset', type=str, metavar='<str>', default='Restaurants',
                    help="Dataset (Laptop/Restaurants) (default=Restaurants)")
parser.add_argument('--glove', dest='glove', type=str, default='data/glove.6B.300d.txt',
                    help='glove path')
parser.add_argument("--mode", dest="mode", type=str, metavar='<str>', default='term',
                    help="Experiment Mode (term|aspect) (default=term)")
parser.add_argument("--mdl", dest="model_type", type=str, metavar='<str>', default='RNN',
                    help="(RNN|TD-RNN|ATT-RNN)")
parser.add_argument("--opt", dest="opt", type=str, metavar='<str>', default='Adam',
                    help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=rmsprop)")
parser.add_argument("--lr", dest='learn_rate', type=float, metavar='<float>', default=0.001,
                    help="Learning Rate")
parser.add_argument("--rnn_type", dest="rnn_type", type=str, metavar='<str>', default='LSTM',
                    help="Recurrent unit type (RNN|LSTM|GRU) (default=LSTM)")
parser.add_argument("--rnn_size", dest="rnn_size", type=int, metavar='<int>', default=300,
                    help="RNN dimension. '0' means no RNN layer (default=300)")
parser.add_argument("--rnn_layers", dest="rnn_layers", type=int, metavar='<int>', default=1,
                    help="Number of RNN layers")
parser.add_argument("--emb_size", dest="embedding_size", type=int, metavar='<int>', default=300,
                    help="Embeddings dimension (default=50)")
parser.add_argument("--batch_size", dest="batch_size", type=int, metavar='<int>', default=256,
                    help="Batch size (default=256)")
parser.add_argument("--pretrained", dest="pretrained", type=int, metavar='<int>', default=1,
                    help="Whether to use pretrained or not")

parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--gpu', dest='gpu', type=int, metavar='<int>', default=0,
                    help="Specify which GPU to use (default=0)")
parser.add_argument('--seed', type=int, default=1111, help='random seed')
args = parser.parse_args()


def init_word_embeddings(word2idx):
    weight = np.random.normal(0, 0.05, [len(word2idx), args.emb_size])
    print('<--loading pre-trained word vectors...-->')
    with open(args.glove, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            content = line.strip().split()
            if content[0] in word2idx:
                weight[word2idx[content[0]]] = np.array(list(map(float, content[1:])))
    return weight


def main():
    data = pickle.load(open('preprocess/data.pkl', 'rb'))
    source_w2i, target_w2i, train, test = data['source_w2i'], data['target_w2i'], data['train'], data['test']
    train['source']
    glove_weight = init_word_embeddings(data['source_w2i'])

    model = BasicRNN(args, len(source_w2i), pretrained=glove_weight)


if __name__ == '__main__':
    main()
