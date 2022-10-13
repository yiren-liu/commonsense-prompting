import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence

class FUDGE_strategy(nn.Module):
    def __init__(self, args, vocab_size):
        super().__init__()
        self.strategy2id = args.strategy2id
        self.tokenizer = args.tokenizer

        self.embedding = nn.Embedding(vocab_size, args.d_model, padding_idx=args.tokenizer.pad_token_id)
        # a LSTM model
        self.rnn = nn.LSTM(args.d_model, args.d_model, num_layers=1, batch_first=True)
        # a linear layer
        self.linear = nn.Linear(args.d_model, len(self.strategy2id))


    def convert_tokenIds_to_strategyIds(self, token_ids):
        """
        token_ids: list of vocab ids
        """
        strategy_ids = []
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        for token in tokens:
            if token in self.strategy2id:
                strategy_ids.append(self.strategy2id[token])
            else:
                strategy_ids.append(self.strategy2id['[None]'])
        return strategy_ids
    

    def forward(self, inputs, lengths=None):
        """
        inputs: token ids, batch x seq, right-padded with 0s
        lengths: lengths of inputs; batch
        future_words: batch x N words to check if not predict next token, else batch
        log_probs: N
        syllables_to_go: batch
        """
        inputs = self.embedding(inputs)
        inputs = pack_padded_sequence(inputs.permute(1, 0, 2), lengths.cpu(), enforce_sorted=False)
        rnn_output, _ = self.rnn(inputs)
        rnn_output, _ = pad_packed_sequence(rnn_output)
        rnn_output = rnn_output.permute(1, 0, 2) # batch x seq x hiddenSize
        return self.linear(rnn_output).squeeze(2)





