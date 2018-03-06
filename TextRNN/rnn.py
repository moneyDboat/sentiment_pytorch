# -*- coding: UTF-8 -*-
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
        self.embed = nn.Embedding(self.vocab_size, self.args.emb_size)
        # rnn_size is the size of hidden state in RNN
        self.rnn = getattr(nn, self.args.rnn_type)(self.args.emb_size, self.args.rnn_size, self.args.rnn_layers,
                                                   bias=False)
        # if 'ATT' in self.args.model_type:
        #     self.attention = AttentionLayer(self.args.rnn_size)
        self.decoder = nn.Linear(self.args.rnn_size, 3)

        self.init_weights(pretrained=pretrained)
        print("<--Initialized {} model-->".format(self.args.rnn_type))

    def forward(self, input, hidden):
        emb = self.embed(input)
        output, hidden = self.rnn(emb, hidden)
        # Attention Layer
        if ('ATT' in self.args.model_type):
            output = self.attention(output, attention_width=self.args.attention_width)
        output = torch.mean(output, 0)
        output = torch.squeeze(output)
        decoded = self.decoder(output)
        # decoded = self.softmax(decoded)
        return decoded, hidden

    def init_weights(self, pretrained):
        initrange = 0.1
        if (pretrained is not None):
            print("Setting Pretrained Embeddings")
            pretrained = pretrained.astype(np.float32)
            pretrained = torch.from_numpy(pretrained)
            # if (self.args.cuda):
            #     pretrained = pretrained.cuda(0)
            self.embed.weight.data = pretrained
        else:
            self.embed.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def repackage_hidden(self, h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        # What's this?
        if type(h) == Variable:
            return Variable(h.data)
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if (self.args.rnn_type == 'LSTM'):
            return (Variable(weight.new(self.args.rnn_layers, bsz, self.args.rnn_size).zero_()),
                    Variable(weight.new(self.args.rnn_layers, bsz, self.args.rnn_size).zero_()))
        else:
            return Variable(weight.new(self.args.rnn_layers, bsz, self.args.rnn_size).zero_())
