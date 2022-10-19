import torch
import torch.nn as nn 
import torch.nn.functional as F

class BART_linear_predictor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.strategy2id = args.strategy2id
        self.tokenizer = args.tokenizer

        # linear layers
        self.linear1 = nn.Linear(args.d_model, args.d_model)
        self.linear2 = nn.Linear(args.d_model, args.d_model)
        self.linear3 = nn.Linear(args.d_model, args.d_model)
        self.linear4 = nn.Linear(args.d_model, len(self.strategy2id))


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
    

    def forward(self, input_hidden_state):
        """
        input_hidden_state: encoder hidden state from BART, batch x hiddenSize
        lengths: lengths of inputs; batch
        output: batch x classes
        """
        x = F.relu(self.linear1(input_hidden_state[:, 0, :]))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        return self.linear4(x)



