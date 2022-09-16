import torch

from transformers import (BartTokenizer,
                            BartForConditionalGeneration, BartConfig)

def getBartTokenizerATOMIC2020(args):
    config = BartConfig.from_pretrained(
        args.model_name_or_path, cache_dir=args.model_cache_dir)
    tokenizer = BartTokenizer.from_pretrained(
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

