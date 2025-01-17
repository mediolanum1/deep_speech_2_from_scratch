
from torch import nn
import torch
import torch.nn.init as init

class CNNLayer(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride, padding = 0):
    super(CNNLayer, self).__init__()
    self.cnn = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    self.batch_norm = nn.BatchNorm2d(num_features = out_channels)
    self.activation = nn.Hardtanh()
    init.xavier_normal_(self.cnn.weight, gain=1) 
    if self.cnn.bias is not None:
        init.constant_(self.cnn.bias, 0)

  def calc_new_sequence_length(self, sequence_lengths): 
    p = self.cnn.padding[1]
    k = self.cnn.kernel_size[1]
    s = self.cnn.stride[1]
    sequence_lengths = torch.tensor(sequence_lengths)
    sequence_lengths = (sequence_lengths + (2*p) - k) // s + 1  
    return torch.clamp(sequence_lengths, min=1)
  
  def forward(self, x, sequence_lengths):
    new_sequence_lengths = self.calc_new_sequence_length(sequence_lengths)
    x = self.cnn(x)
    x = self.batch_norm(x)
    x = self.activation(x)
    return x, new_sequence_lengths
  
class RNNLayer(nn.Module):
  def __init__(self, in_channels, hidden_units):
    super(RNNLayer, self).__init__()
    self.rnn = nn.LSTM(in_channels, hidden_units, bidirectional = True, batch_first = True)
    self.dropout = nn.Dropout(p=0.3)
    self.layer_norm = nn.LayerNorm(hidden_units * 2) 
    self.activation = nn.Hardtanh()
    self._apply_xavier_initialization()

  def _apply_xavier_initialization(self):
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:  
                init.xavier_normal_(param, gain=1)  
            elif 'weight_hh' in name: 
                init.xavier_normal_(param, gain=1)
            elif 'bias' in name:  
                init.constant_(param, 0)

  def forward(self, x, sequence_lengths):
    input_packed = nn.utils.rnn.pack_padded_sequence(x, sequence_lengths.cpu(), enforce_sorted=False, batch_first=True) # ok here we prepare input by removing 0-padding, for it we need                                                                                     # CPU
    try: 
            output, hidden_states = self.rnn(input_packed)
    except RuntimeError as e:
            print("Input Packed:", x)
            raise e
    x, _ = nn.utils.rnn.pad_packed_sequence(output, total_length=x.shape[1],batch_first=True)
    x = self.dropout(x)
    x = self.layer_norm(x)
    x = self.activation(x)
    return x


class DeepSpeech2(nn.Module):
  def __init__(self, input_features):
    super(DeepSpeech2, self).__init__()
    self.cnn_layers = nn.ModuleList([ # notice, its not ModuleSequential, it allows for more complex forward passes or smth
        CNNLayer(1,32, (11,11), (2,2)),
        CNNLayer(32,32, (11,11), (1,1), padding=(5,0)),
        CNNLayer(32,64, (11,11), (1,1), padding=(5,0)),
    ]   
    )
        # Start with TaskConfig.n_mels and apply each CNN layer's transformations
    features = (input_features - 11) // 2 + 1  # First CNN layer
    # Then, multiply this final result by the output channels of the last CNN layer (96 here).
    self.features = features
    self.rnn_layers = nn.ModuleList([
        RNNLayer(
            in_channels = 64 * self.features if i == 0 else 330 * 2,
            hidden_units = 330
        ) for i in range(3)
    ])
    # classifier layer
    self.classifier = nn.Sequential(
        nn.Linear(330 * 2, 28), # we multiply by 2 since its biderectional and we have 2 hidden states
    )

  def forward(self,x, sequence_lengths):
    x = x.unsqueeze(1) # (batch_size, 1, features, time)
    for cnn_layer in self.cnn_layers:
          x, sequence_lengths =    cnn_layer(x, sequence_lengths)
    x = x.view(x.shape[0], x.shape[1] * x.shape[2] , x.shape[3]) # (batch_size, 96 * n_features, time)
    x = x.transpose(1,2)                                         # (batch_size, time, 96 * n_features)   
    for rnn_layer in self.rnn_layers:
          x = rnn_layer(x, sequence_lengths) # (batch_size, time, 2 * hidden_units)
    x = self.classifier(x) # (batch_size, time, vocab_size)
    return {"log_probs": nn.functional.log_softmax(x, dim=-1), "log_probs_length": sequence_lengths}
