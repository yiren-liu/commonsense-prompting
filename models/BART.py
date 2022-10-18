import torch

from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
)

from utils.fudge_utils import predict_formality

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
        pass

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
    
    def generate_strategy(self, input_ids, next_strategy_id, use_gts=False, **kwargs):
        # input_ids: [batch_size, seq_len]
        # next_strategy_id: [batch_size, 1]
        # use_gts: whether to use ground truth strategy

        if use_gts:
            return next_strategy_id


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