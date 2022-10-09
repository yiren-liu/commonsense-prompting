import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence

class FUDGE(nn.Module):
    def __init__(self, args, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, args.d_model, padding_idx=args.pad_token_id)
        # a LSTM model
        self.rnn = nn.LSTM(args.d_model, args.d_model, num_layers=1, batch_first=True)
        # a linear layer
        self.linear = nn.Linear(args.d_model, 1)
    
    def forward(self, inputs, lengths=None):
        inputs = self.marian_embed(inputs)
        inputs = pack_padded_sequence(inputs.permute(1, 0, 2), lengths.cpu(), enforce_sorted=False)
        rnn_output, _ = self.rnn(inputs)
        rnn_output, _ = pad_packed_sequence(rnn_output)
        rnn_output = rnn_output.permute(1, 0, 2) # batch x seq x hiddenSize
        return self.out_linear(rnn_output).squeeze(2)





