import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class GRUD(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 gru_dropout=0.3, decoder_dropout=0.5, batch_first=True):
        super(GRUD, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_first = batch_first

        # initialize weights and biases
        self.W_r = nn.Parameter(torch.FloatTensor(input_size, hidden_size).normal_(0, 0.02))
        self.U_r = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size).normal_(0, 0.02))
        self.V_r = nn.Parameter(torch.FloatTensor(input_size, hidden_size).normal_(0, 0.02))
        self.b_r = nn.Parameter(torch.FloatTensor(hidden_size).zero_())

        self.W_z = nn.Parameter(torch.FloatTensor(input_size, hidden_size).normal_(0, 0.02))
        self.U_z = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size).normal_(0, 0.02))
        self.V_z = nn.Parameter(torch.FloatTensor(input_size, hidden_size).normal_(0, 0.02))
        self.b_z = nn.Parameter(torch.FloatTensor(hidden_size).zero_())

        self.W = nn.Parameter(torch.FloatTensor(input_size, hidden_size).normal_(0, 0.02))
        self.U = nn.Parameter(torch.FloatTensor(hidden_size, hidden_size).normal_(0, 0.02))
        self.V = nn.Parameter(torch.FloatTensor(input_size, hidden_size).normal_(0, 0.02))
        self.b = nn.Parameter(torch.FloatTensor(hidden_size).zero_())

        # since W_gamma_x is diagonal, just initialize 1-d
        self.W_gamma_x = nn.Parameter(torch.FloatTensor(input_size).normal_(0, 0.02))
        self.b_gamma_x = nn.Parameter(torch.FloatTensor(input_size).zero_())

        self.W_gamma_h = nn.Parameter(torch.FloatTensor(input_size, hidden_size).normal_(0, 0.02))
        self.b_gamma_h = nn.Parameter(torch.FloatTensor(hidden_size).zero_())
        self.gru_dropout = gru_dropout

        self.decoder = nn.Linear(hidden_size, output_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.decoder_dropout = nn.Dropout(p=decoder_dropout)

    def forward(self, x, delta, m, x_forward, batch_sizes, h_t=None):
        """

        :param x: features input [batch_size, seq_len, num_features]
        :param delta: time interval of feature observation [batch_size, seq_len, num_features]
        :param m: masking vector {0, 1} of [batch_size, seq_len, num_features]
        :param x_forward: place to replace missing values with [b, seqlen, numf]
        :param h_t: initial hidden state [batch_size, seq_len, hidden_size]
        :return: output [batch_size, output_size], hidden_state [batch_size, hidden_size]
        """
        batch_size, seq_len, input_size = x.size()
        if h_t is None:
            # initialize to zero
            h_t = Variable(torch.FloatTensor(batch_size, self.hidden_size).zero_())
            if x.is_cuda:
                h_t = h_t.cuda()

        # compute decays
        decay_x = delta * self.W_gamma_x + self.b_gamma_x
        zeroes = Variable(torch.zeros(decay_x.size()))
        if decay_x.is_cuda:
            zeroes = zeroes.cuda()
        gamma_x_t = torch.exp(-torch.max(zeroes, decay_x))

        decay_h = torch.matmul(m, self.W_gamma_h) + self.b_gamma_h
        zeroes = Variable(torch.zeros(decay_h.size()))
        if decay_h.is_cuda:
            zeroes = zeroes.cuda()
        gamma_h_t = torch.exp(-torch.max(zeroes, decay_h))

        # replace missing values
        x_replace = gamma_x_t * x_forward + (1 - gamma_x_t) * 0.001
        x[m.byte()] = x_replace[m.byte()]

        # dropout masks, one for each batch
        dropout_rate = self.gru_dropout if self.training else 0.

        W_dropout = Variable((torch.FloatTensor(self.W.size()).uniform_() > dropout_rate).float())
        U_dropout = Variable((torch.FloatTensor(self.U.size()).uniform_() > dropout_rate).float())
        V_dropout = Variable((torch.FloatTensor(self.V.size()).uniform_() > dropout_rate).float())

        if decay_h.is_cuda:
            W_dropout = W_dropout.cuda()
            U_dropout = U_dropout.cuda()
            V_dropout = V_dropout.cuda()

        for t in range(seq_len):
            # decay h
            update_range = Variable(torch.LongTensor(list(range(batch_sizes[t]))))
            if decay_h.is_cuda:
                update_range = update_range.cuda()
            h_t = h_t.clone().index_copy_(0, update_range, gamma_h_t[:batch_sizes[t], t, :] * h_t[:batch_sizes[t]])

            z_t = F.sigmoid(torch.matmul(x[:batch_sizes[t], t, :], self.W_z) + torch.matmul(h_t[:batch_sizes[t]],
                                                                                            self.U_z) + torch.matmul(
                1 - m[:batch_sizes[t], t, :], self.V_z) + self.b_z)
            r_t = F.sigmoid(torch.matmul(x[:batch_sizes[t], t, :], self.W_r) + torch.matmul(h_t[:batch_sizes[t]],
                                                                                            self.U_r) + torch.matmul(
                1 - m[:batch_sizes[t], t, :], self.V_r) + self.b_r)
            # h_tilde_t = F.tanh(torch.matmul(x[:batch_sizes[t], t, :], self.W) + torch.matmul(h_t[:batch_sizes[t]] * r_t,
            #                                                                                  self.U) + torch.matmul(
            #     1 - m[:batch_sizes[t], t, :], self.V) + self.b)
            h_tilde_t = F.tanh(torch.matmul(x[:batch_sizes[t], t, :], self.W*W_dropout) + torch.matmul(h_t[:batch_sizes[t]]*r_t, self.U*U_dropout) + torch.matmul(1-m[:batch_sizes[t], t, :], self.V*V_dropout) + self.b)
            h_t = h_t.clone().index_copy_(0, update_range, (1 - z_t) * h_t[:batch_sizes[t]] + z_t * h_tilde_t)

        if batch_size > 1:
            h_t = self.bn(h_t)

        output = F.log_softmax(self.decoder(self.decoder_dropout(h_t)), dim=-1)

        return output, h_t

