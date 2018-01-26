import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class BasicRNN(nn.Module):
    """Container module with an encoder(embedding), a recurrent module, and a decoder."""

    def __init__(self, args, vocab_size, pretrained=None):
        super(BasicRNN, self).__init__()
        self.args = args
        self.vocab_size = vocab_size
        self.encoder = nn.Embedding(self.vocab_size, self.args.embedding_size)
        # rnn_size is the size of hidden state in RNN
        self.rnn = getattr(nn, self.args.rnn_type)(self.args.embedding_size, self.args.rnn_size, self.args.rnn_layers,
                                                   bias=False)
        self.decoder = nn.Linear(self.args.rnn_size, 3)
        self.softmax = nn.Softmax(dim=0)
        self.init_weights(pretrained=pretrained)
        print("Initialized {} model".format(self.args.rnn_type))

    def init_weights(self, pretrained):
        initrange = 0.1
        if (pretrained is not None):
            print("Setting Pretrained Embeddings")
            pretrained = pretrained.astype(np.float32)
            pretrained = torch.from_numpy(pretrained)
            if (self.args.cuda):
                pretrained = pretrained.cuda()
            self.encoder.weight.data = pretrained
        else:
            self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if (self.args.rnn_type == 'LSTM'):
            return (Variable(weight.new(self.args.rnn_layers, bsz, self.args.rnn_size).zero_()),
                    Variable(weight.new(self.args.rnn_layers, bsz, self.args.rnn_size).zero_()))
        else:
            return Variable(weight.new(self.args.rnn_layers, bsz, self.args.rnn_size).zero_())

    def forward(self, input, hidden):
        emb = self.encoder(input)
        output, hidden = self.rnn(emb, hidden)
        last = Variable(torch.LongTensor([output.size()[0] - 1]))
        if (self.args.cuda):
            last = last.cuda()

        # Attention Layer
        if ('ATT' in self.args.model_type):
            output = self.AttentionLayer(output, attention_width=self.args.attention_width)
        if (self.args.aggregation == 'mean'):
            output = torch.mean(output, 0)
        elif (self.args.aggregation == 'last'):
            output = torch.index_select(output, 0, last)

        output = torch.squeeze(output)
        decoded = self.decoder(output)
        decoded = self.softmax(decoded)
        return decoded, hidden

    def train(self, mode=True):
        pass
