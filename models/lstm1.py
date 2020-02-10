import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class LSTM1(torch.nn.Module):
    def __init__(self, inputSize, outputSize, lstms_in_out=((10, 50), (10, 50)), linear_layers=(10,10), out_size=1, hidden_activation=nn.ReLU, final_activation=nn.ReLU):
        super(LSTM1, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

        self.eng_lstm = nn.LSTM(*lstms_in_out[0])  # Input dim is 3, output dim is 3
        self.ger_lstm = nn.LSTM(*lstms_in_out[1])
        self.eng_hidden = (torch.randn(1, 1, 10), torch.randn(1, 1, 10))
        self.ger_hidden = (torch.randn(1, 1, 10), torch.randn(1, 1, 10))

        self.linears = [nn.Linear(i, o) for i, o in zip([(lstms_in_out[0][1] + lstms_in_out[1][1])]+linear_layers[:], linear_layers[:]+[out_size])]

    def forward(self, eng, ger):
        eng_out, eng_hidden = eng_lstm(eng, self.eng_hidden)
        ger_out, ger_hidden = ger_lstm(ger, self.ger_hidden)

        cat = torch.cat((eng_hidden[0], ger_hidden[0]), 0)

        #TODO:  FINISH ME
        out = self.linear(x)
        return out

torch.manual_seed(1)

eng_lstm = nn.LSTM(10, 50)  # Input dim is 3, output dim is 3
ger_lstm = nn.LSTM(10, 50)

eng_inputs = [torch.randn(1, 10) for _ in range(5)]  # make a sequence of length 5
ger_inputs = [torch.randn(1, 10) for _ in range(6)]  # make a sequence of length 5


eng_inputs = torch.cat(eng_inputs).view(len(eng_inputs), 1, -1)
ger_inputs = torch.cat(ger_inputs).view(len(ger_inputs), 1, -1)

# initialize the hidden state.
eng_hidden = (torch.randn(1, 1, 10), torch.randn(1, 1, 10))
ger_hidden = (torch.randn(1, 1, 10), torch.randn(1, 1, 10))

eng_out, eng_hidden = eng_lstm(eng_inputs, eng_hidden)
ger_out, ger_hidden = ger_lstm(ger_inputs, ger_hidden)


lin = nn.Linear(100, 1)
lin(eng_hidden + ger_hidden)
# alternatively, we can do the entire sequence all at once.
# the first value returned by LSTM is all of the hidden states throughout
# the sequence. the second is just the most recent hidden state
# (compare the last slice of "out" with "hidden" below, they are the same)
# The reason for this is that:
# "out" will give you access to all hidden states in the sequence
# "hidden" will allow you to continue the sequence and backpropagate,
# by passing it as an argument  to the lstm at a later time
# Add the extra 2nd dimension

print(out)
print(hidden)

