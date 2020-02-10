import torch
import torch.nn as nn


class LSTM1(torch.nn.Module):
    def __init__(self, lstms_in_out=((10, 50), (10, 50)), linear_layers=(10,10), out_size=1, hidden_activation=nn.ReLU, final_activation=nn.ReLU):
        super(LSTM1, self).__init__()
        self.final_activation = final_activation
        self.hidden_activation = hidden_activation

        self.eng_lstm = nn.LSTM(*lstms_in_out[0])  # Input dim is 3, output dim is 3
        self.ger_lstm = nn.LSTM(*lstms_in_out[1])
        concat_size = lstms_in_out[0][1] + lstms_in_out[1][1]
        self.linears = [nn.Linear(i, o) for i, o in zip([concat_size]+linear_layers[:], linear_layers[:]+[out_size])]

    def forward(self, eng, ger):
        eng_out, eng_hidden = self.eng_lstm(eng)
        ger_out, ger_hidden = self.ger_lstm(ger)

        cat = torch.cat((eng_hidden[0], ger_hidden[0]), 0)

        val = cat
        for l in self.linears:
            val = self.hidden_activation()(val)
            val = l(val)

        out = self.final_activation()(val)
        return out



