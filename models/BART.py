import os
import torch

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
)

from utils.fudge_utils import predict_formality
from models.strategy_predictor.Linear import BART_linear_predictor

def getBartTokenizerATOMIC2020(args):
    tokenizer = BartTokenizer.from_pretrained(
        args.model_name_or_path, cache_dir=args.model_cache_dir)
    # add special tokens cls_token
    tokenizer.add_special_tokens({'cls_token': '<s>'})
    additional_special_tokens = [
        "[Question]", "[Reflection of feelings]", "[Information]", "[Restatement or Paraphrasing]",
        "[Others]", "[Self-disclosure]", "[Affirmation and Reassurance]", "[Providing Suggestions]",
        "[None]"
    ]
    tokenizer.add_tokens(additional_special_tokens)
    return tokenizer


class BartATOMIC2020(BartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.strategy_tokens = [
            "[Question]", "[Reflection of feelings]", "[Information]", "[Restatement or Paraphrasing]",
            "[Others]", "[Self-disclosure]", "[Affirmation and Reassurance]", "[Providing Suggestions]",
            "[None]"
        ]
        self.strategy_classifier = None
        self.strategy_criterion = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # model(input_ids, attention_mask=input_ids.ne(tokenizer.pad_token_id),
        # decoder_input_ids=decoder_input_ids, decoder_turn_ids=decoder_turn_ids,
        # decoder_role_ids=decoder_role_ids, turn_ids=turn_ids,
        # role_ids=role_ids, labels=decoder_label_ids,
        # decoder_strategy_ids=decoder_strategy_ids,
        # comet_embs=comet_embs, comet_mask=comet_mask,
        # comet_embs_st=comet_embs_st, comet_mask_st=comet_mask_st, emotion=emotion)
        # raise NotImplementedError("BartATOMIC2020 is not implemented yet.")
        output = super().forward(
            input_ids=input_ids, attention_mask=attention_mask,
            #     decoder_input_ids=decoder_input_ids,
            #     decoder_attention_mask=decoder_attention_mask,
            labels=labels, **kwargs
        )
        return output

    def train_with_classifier_loss(self, input_ids, attention_mask, labels, next_strategy_id, args, **kwargs):
        if not self.strategy_classifier:
            print("Strategy classifier is not initialized.")
            print("initializing strategy classifier...")
            self.strategy_classifier = BART_linear_predictor(args).to(args.device)
        # mask the first token of labels, which is the strategy token
        labels = labels.clone()
        labels[:, 0] = -100
        output_lm = self(
            input_ids=input_ids, attention_mask=attention_mask, 
            labels=labels, 
            output_hidden_states=True, **kwargs
        )
        
        output_classifier = self.strategy_classifier(output_lm.encoder_last_hidden_state)
        self.strategy_criterion = self.strategy_criterion.to(args.device)
        
        tgt_ids = self.convert_tokenIds_to_strategyIds(next_strategy_id, args)
        tgt_ids = torch.LongTensor(tgt_ids).to(args.device)
        loss_classifier = self.strategy_criterion(output_classifier, tgt_ids)
        return output_lm, loss_classifier

    def generate_strategy(self, input_ids, next_strategy_id, args, **kwargs):
        # input_ids: [batch_size, seq_len]
        # next_strategy_id: [batch_size, 1]
        # use_gts: whether to use ground truth strategy
        if args.strategy_predictor == "gts":
            return next_strategy_id
        elif args.strategy_predictor == "lm":
            # generate from a list of oracle strategies given the input
            strategy_ids = args.tokenizer.convert_tokens_to_ids(self.strategy_tokens)
            with torch.no_grad():
                # get the logits of the next token
                outputs = self(input_ids=input_ids, **kwargs)
                next_token_logits = outputs[0][:, -1, :]
                # get the logits of the strategy tokens
                strategy_logits = next_token_logits[:, strategy_ids]
                # get the max strategy logits
                max_strategy_logits, max_strategy_ids = torch.max(strategy_logits, dim=1)
                best_strategy_ids = strategy_ids[max_strategy_ids]
            return torch.LongTensor([best_strategy_ids]).to(args.device)
        elif args.strategy_predictor == "classifier":
            # use a classifier to predict the strategy
            if not self.strategy_classifier:
                if not os.path.exists(args.load_dir + "/strategy_classifier.bin"):
                    raise ValueError("Classifier model does not exist. Please train it first.")
                classifier = torch.load(args.load_dir + "/strategy_classifier.bin").to(args.device)
            strategy_ids = args.tokenizer.convert_tokens_to_ids(self.strategy_tokens)

            with torch.no_grad():
                output_lm = self(
                    input_ids=input_ids,
                    output_hidden_states=True, **kwargs
                )
                classifier_logits = classifier(output_lm.encoder_last_hidden_state)
                max_strategy_logits, max_strategy_ids = torch.max(classifier_logits, dim=1)
                best_strategy_ids = strategy_ids[max_strategy_ids]
            # raise NotImplementedError
            return torch.LongTensor([best_strategy_ids]).to(args.device)
        else:
            raise NotImplementedError



    def generate_fudge(self, encoder_input_ids, model, tokenizer, conditioning_model, target_attr_idx, decoder_start_token_id=None, precondition_topk=200, length_cutoff=512, condition_lambda=1.0, device='cuda'):
        results = predict_formality(
            model,
            tokenizer,
            conditioning_model,
            target_attr_idx,
            encoder_input_ids,
            decoder_start_token_id=decoder_start_token_id,
            precondition_topk=precondition_topk,
            do_sample=False,
            length_cutoff=length_cutoff,
            condition_lambda=condition_lambda,
            device=device
        )
        return results


    def convert_tokenIds_to_strategyIds(self, token_ids, args):
        """
        token_ids: list of vocab ids
        """
        strategy_ids = []
        tokens = args.tokenizer.convert_ids_to_tokens(token_ids)
        for token in tokens:
            if token in args.strategy2id:
                strategy_ids.append(args.strategy2id[token])
            else:
                strategy_ids.append(args.strategy2id['[None]'])
        return strategy_ids