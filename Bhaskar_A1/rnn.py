import torch
import torch.nn as nn
import torch.functional as F

class MyRnn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyRnn, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.softmax2 = nn.LogSoftmax(dim=0)

    def forward(self, input, hidden):
        if len(input.size())==1:
            combined=torch.cat((input, hidden))
        else:
            combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        if len(input.size())==1:
            output = self.softmax2(output)
        else:
            output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return nn.init.kaiming_uniform_(torch.empty(1, self.hidden_size))

