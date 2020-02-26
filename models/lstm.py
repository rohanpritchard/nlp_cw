import torch
import torch.nn as nn


class LSTM(nn.Module):
  def __init__(self, a_vocab_size=None, b_vocab_size=None, padding_index=0, lstms_in_out=((5, 5), (5, 5)),
               linear_layers=(10, 5), out_size=1, hidden_activation=nn.ReLU, final_activation=None):
    super(LSTM, self).__init__()
    if a_vocab_size is not None:
      self.a_embedding_layer = nn.Embedding(a_vocab_size, lstms_in_out[0][0], padding_idx=padding_index)
    else:
      self.a_embedding_layer = None

    if b_vocab_size is not None:
      self.b_embedding_layer = nn.Embedding(b_vocab_size, lstms_in_out[1][0], padding_idx=padding_index)
    else:
      self.b_embedding_layer = None

    self.final_activation = final_activation
    self.hidden_activation = hidden_activation

    self.a_lstm = nn.LSTM(*lstms_in_out[0], batch_first=True)
    self.b_lstm = nn.LSTM(*lstms_in_out[1], batch_first=True)
    self.concat_size = lstms_in_out[0][1] + lstms_in_out[1][1]
    self.linears = [nn.Linear(i, o) for i, o in
                    zip([self.concat_size] + list(linear_layers), list(linear_layers) + [out_size])]

  def forward(self, a, a_len, b, b_len):

    a_embedded = a if self.a_embedding_layer is None else self.a_embedding_layer(a)
    b_embedded = b if self.b_embedding_layer is None else self.b_embedding_layer(b)

    # Ignore the embeddings of zeros corresponding to the end of the sentence.
    a_embedded_chopped = torch.nn.utils.rnn.pack_padded_sequence(a_embedded, a_len, batch_first=True,
                                                                   enforce_sorted=False)
    b_embedded_chopped = torch.nn.utils.rnn.pack_padded_sequence(b_embedded, b_len, batch_first=True,
                                                                   enforce_sorted=False)

    a_out, a_hidden = self.a_lstm(a_embedded_chopped)
    b_out, b_hidden = self.b_lstm(b_embedded_chopped)

    cat = torch.cat((a_hidden[0], b_hidden[0]), -1).view(-1, self.concat_size)

    val = cat
    for l in self.linears:
      val = self.hidden_activation()(val)
      val = l(val)

    if self.final_activation is not None:
      out = self.final_activation()(val)
    else:
      out = val
    return out