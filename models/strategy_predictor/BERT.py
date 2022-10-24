import torch
import torch.nn as nn 
import torch.nn.functional as F

from transformers import (
    RobertaForSequenceClassification,
    RobertaConfig,
    RobertaTokenizer,
)

STRATEGYLIST = [
    "[Question]", "[Reflection of feelings]", "[Information]", "[Restatement or Paraphrasing]",
    "[Others]", "[Self-disclosure]", "[Affirmation and Reassurance]", "[Providing Suggestions]",
    "[None]"
]

class BERT_predictor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.strategy2id = args.strategy2id
        self.tokenizer = args.tokenizer

        if args.pretrained_predictor_dir:
            self.bert = torch.load(args.pretrained_predictor_dir + "/pytorch_model.bin").to(args.device)
            args.tokenizer = RobertaTokenizer.from_pretrained(
                args.pretrained_predictor_dir, cache_dir=args.model_cache_dir
            )
            pass
        else:
            self.bert = RobertaForSequenceClassification.from_pretrained(
                'roberta-base', num_labels=len(self.strategy2id), cache_dir=args.model_cache_dir
            ).to(args.device)
            args.tokenizer = RobertaTokenizer.from_pretrained(
                'roberta-base', cache_dir=args.model_cache_dir
            )
            additional_special_tokens = [
                "[Question]", "[Reflection of feelings]", "[Information]", "[Restatement or Paraphrasing]",
                "[Others]", "[Self-disclosure]", "[Affirmation and Reassurance]", "[Providing Suggestions]",
                "[None]"
            ]
            args.tokenizer.add_tokens(additional_special_tokens)
            self.bert.resize_token_embeddings(len(args.tokenizer))
        
    
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
        output: batch x classes
        """
        outputs = self.bert(inputs[:, :512])
        if isinstance(outputs, torch.Tensor):
            logits = outputs
        else:
            logits = outputs.logits
        return logits
