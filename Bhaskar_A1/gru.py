import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable
import numpy as np
class MyGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=True)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.i2o = nn.Linear(hidden_size, output_size, bias = True)
        self.softmax = nn.LogSoftmax(dim=1)
        self.softmax2 = nn.LogSoftmax(dim=0)
        self.reset_parameters()


    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, input, hx=None):

        if hx is None:
            hx = Variable(input.new_zeros(input.size(0), self.hidden_size))

        x_t = self.x2h(input)
        h_t = self.h2h(hx)

        if len(input.size())==1:

            x_reset, x_upd, x_new = x_t.chunk(3, 0)
            h_reset, h_upd, h_new = h_t.chunk(3, 0)
        else:
            x_reset, x_upd, x_new = x_t.chunk(3, 1)
            h_reset, h_upd, h_new = h_t.chunk(3, 1)

        reset_gate = torch.sigmoid(x_reset + h_reset)
        update_gate = torch.sigmoid(x_upd + h_upd)
        new_gate = torch.tanh(x_new + (reset_gate * h_new))

        hy = update_gate * hx + (1 - update_gate) * new_gate
        output = self.i2o(hy)
        if len(input.size())==1:
            output = self.softmax2(output)
        else:
            output = self.softmax(output)
        return output, hy