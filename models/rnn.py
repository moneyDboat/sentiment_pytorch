import torch
import torch.nn as nn
import time
import random
from torch.autograd import Variable
import numpy as np


class BasicRNN(nn.Module):
    """Container module with an encoder(embedding), a recurrent module, and a decoder."""

    def __init__(self, args, vocab_size, pretrained=None):
        super(BasicRNN, self).__init__()
        self.args = args
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(self.vocab_size, self.args.embedding_size)
        # rnn_size is the size of hidden state in RNN
        self.rnn = getattr(nn, self.args.rnn_type)(self.args.embedding_size, self.args.rnn_size, self.args.rnn_layers,
                                                   bias=False)
        self.decoder = nn.Linear(self.args.rnn_size, 3)
        self.softmax = nn.Softmax(dim=0)

        self.init_weights(pretrained=pretrained)
        print("<--Initialized {} model-->".format(self.args.rnn_type))

    def forward(self, input, hidden):
        emb = self.encoder(input)
        output, hidden = self.rnn(emb, hidden)
        output = torch.mean(output, 0)

        # # Attention Layer
        # if ('ATT' in self.args.model_type):
        #     output = self.AttentionLayer(output, attention_width=self.args.attention_width)

        output = torch.squeeze(output)
        decoded = self.decoder(output)
        decoded = self.softmax(decoded)
        return decoded, hidden

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

    # def init_hidden(self, bsz):
    #     weight = next(self.parameters()).data
    #     if (self.args.rnn_type == 'LSTM'):
    #         return (Variable(weight.new(self.args.rnn_layers, bsz, self.args.rnn_size).zero_()),
    #                 Variable(weight.new(self.args.rnn_layers, bsz, self.args.rnn_size).zero_()))
    #     else:
    #         return Variable(weight.new(self.args.rnn_layers, bsz, self.args.rnn_size).zero_())

    def train(self, mode=True):
        self.criterion = nn.CrossEntropyLoss()

        total_loss = 0
        num_batches = int(len(self.train_set) / self.args.batch_size) + 1
        self.select_optimizer()

        print("<---Starting training--->")
        for epoch in range(1, self.args.epochs + 1):
            t0 = time.clock()
            random.shuffle(self.train_set)
            print("========================================================================")
            losses = []

            for i in range(num_batches):
                loss = 0
                if (self.args.model_type in ['TD-RNN']):
                    loss = self.train_target_batch(i)
                else:
                    loss = self.train_batch(i)
                if (loss is None):
                    continue
                losses.append(loss)

            t1 = time.clock()
            print("[Epoch {}] Train Loss={} T={}s".format(epoch, np.mean(losses), t1 - t0))
            if (epoch > 0 and epoch % self.args.eval == 0):
                if (self.args.model_type in ['TD-RNN']):
                    self.evaluate_target(self.test_set)
                else:
                    self.evaluate(self.test_set)

    def evaluate(self):
        pass

    def get_accuracy(self):
        pass

    def pad_to_batch_max(self):
        pass
