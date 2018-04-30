import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class GRU_decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 gru_dropout=0.3, decoder_dropout=0.5, batch_first=True):
        super(GRU_decoder, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size,
                          batch_first=batch_first, dropout=gru_dropout)
        self.decoder = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            #                                 nn.Dropout(decoder_dropout),
            nn.Linear(hidden_size, output_size),
            nn.LogSoftmax()
        )

    def forward(self, x):
        _, hidden = self.gru(x)
        output = self.decoder(hidden.squeeze()).squeeze()

        return output


class GRU_D(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 gru_dropout=0.3, decoder_dropout=0.5, batch_first=True):
        super(GRUD, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_first = batch_first
        #         self.feature_means = Variable(torch.FloatTensor(feature_means), requires_grad=False)
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

    def forward(self, x, delta, m, x_forward, h_t=None):
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

        def hasnan(x):
            return (x != x).any()

        # compute decays
        decay_x = delta * self.W_gamma_x + self.b_gamma_x
        zeroes = Variable(torch.zeros(decay_x.size()))
        if decay_x.is_cuda:
            zeroes = zeroes.cuda()
        gamma_x_t = torch.exp(-torch.max(zeroes, decay_x))
        if hasnan(gamma_x_t):
            print("gamma_x_t", hasnan(decay_x), hasnan(self.W_gamma_x), hasnan(self.b_gamma_x))

        decay_h = torch.matmul(m, self.W_gamma_h) + self.b_gamma_h
        zeroes = Variable(torch.zeros(decay_h.size()))
        if decay_h.is_cuda:
            zeroes = zeroes.cuda()
        gamma_x_h = torch.exp(-torch.max(zeroes, decay_h))
        if hasnan(gamma_x_h):
            print("gamma_x_h", hasnan(decay_h), hasnan(m), hasnan(self.W_gamma_h), hasnan(self.b_gamma_h))

        # replace missing values
        # TODO: empirical mean is just 0???
        x_replace = decay_x * x_forward + (1 - decay_x) * 0.001
        x[m.byte()] = x_replace[m.byte()]
        if hasnan(x):
            print("x", hasnan(x_replace), hasnan(x_forward))

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
            h_t = gamma_x_h[:, t, :] * h_t
            z_t = F.sigmoid(
                torch.matmul(x[:, t, :], self.W_z) + torch.matmul(h_t, self.U_z) + torch.matmul(1 - m[:, t, :],
                                                                                                self.V_z) + self.b_z)
            r_t = F.sigmoid(
                torch.matmul(x[:, t, :], self.W_r) + torch.matmul(h_t, self.U_r) + torch.matmul(1 - m[:, t, :],
                                                                                                self.V_r) + self.b_r)
            h_tilde_t = F.tanh(torch.matmul(x[:, t, :], self.W * W_dropout) + torch.matmul(h_t * r_t,
                                                                                           self.U * U_dropout) + torch.matmul(
                1 - m[:, t, :], self.V * V_dropout) + self.b)
            h_t = (1 - z_t) * h_t + z_t * h_tilde_t

        if batch_size > 1:
            h_t = self.bn(h_t)
        if hasnan(h_t):
            print("h_t")
            print()
        output = F.log_softmax(self.decoder(self.decoder_dropout(h_t)), dim=-1)

        return output, h_t


if __name__ == "__main__":
    print("GRU_D")

    model = GRUD(100, 50, 2)

    x = Variable(torch.FloatTensor(128, 20, 100).normal_())
    delta = Variable(torch.FloatTensor(x.size()).normal_())
    m = Variable(torch.FloatTensor(x.size()).normal_())

    print(model(x, delta, m))