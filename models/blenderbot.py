import torch

from transformers import (BlenderbotSmallTokenizer,
                          BlenderbotSmallForConditionalGeneration, BlenderbotSmallConfig)


def getBlenderbotTokenizerATOMIC2020(args):
    config = BlenderbotSmallConfig.from_pretrained(
        args.model_name_or_path, cache_dir=args.model_cache_dir)
    tokenizer = BlenderbotSmallTokenizer.from_pretrained(
        args.model_name_or_path, cache_dir=args.model_cache_dir)

    # load tokenizer
    additional_special_tokens = ["[Question]", "[Reflection of feelings]", "[Information]", "[Restatement or Paraphrasing]",
                                 "[Others]", "[Self-disclosure]", "[Affirmation and Reassurance]", "[Providing Suggestions]"]
    # comet_additional_special_tokens = ["[xAttr]", "[xEffect]", "[xIntent]","[xNeed]", "[xReact]", "[xWant]"]
    comet_additional_special_tokens = ["[xAttr]", "[xEffect]", "[xIntent]",
                                       "[xNeed]", "[xReact]", "[xWant]", "[oWant]", "[oEffect]", "[oReact]"]
    tokenizer.add_tokens(additional_special_tokens)
    tokenizer.add_tokens(comet_additional_special_tokens)
    tokenizer.add_special_tokens({'cls_token': '[CLS]'})
    return tokenizer


class BlenderbotATOMIC2020(BlenderbotSmallForConditionalGeneration):
    def __init__(self, config: BlenderbotSmallConfig):
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
        output = super().forward(
            input_ids=input_ids, attention_mask=attention_mask,
        #     decoder_input_ids=decoder_input_ids,
        #     decoder_attention_mask=decoder_attention_mask,
            labels=labels, **kwargs
        )
        return output

#     def generate(self, **kwargs):
#         # print("inputs", inputs)
#         return super().generate(**kwargs)
