import torch
import torch.nn as nn


class RNNFromScratch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNFromScratch, self).__init__()

        self.hidden_size = hidden_size
        self.W_hx = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size)
        self.W_oh = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        W_hx_x = self.W_hx(input)
        hidden = self.W_hh(hidden + W_hx_x)
        output = self.W_oh(hidden)
        y_hat = self.softmax(output)

        return y_hat, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

class RNNFromScratchMultiLayer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layer_size):
        super(RNNFromScratchMultiLayer, self).__init__()
        pass



class RNNUsingModule(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNUsingModule, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.bias = nn.Parameter(torch.ones([output_size]))

    def forward(self, input, hidden):
        output, _ = self.rnn(input, hidden)
        output = output[:, -1, :]
        output = self.fc(output) + self.bias
        return output

    def initHidden(self):
        return torch.zeros([1, 1, self.hidden_size])
        