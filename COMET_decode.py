import json
import os 
import torch
import argparse

import numpy as np
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


from utils.comet_utils import use_task_specific_params, trim_batch

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class Comet:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        task = "summarization"
        use_task_specific_params(self.model, task)
        # self.batch_size = 2
        self.batch_size = 20
        self.decoder_start_token_id = None

    def generate(
            self, 
            queries,
            decode_method="beam", 
            num_generate=5, 
            ):

        with torch.no_grad():
            examples = queries

            decs = []
            for batch in tqdm(list(chunks(examples, self.batch_size))):

                batch = self.tokenizer(batch, return_tensors="pt", truncation=True, padding="max_length").to(self.device)
                input_ids, attention_mask = trim_batch(**batch, pad_token_id=self.tokenizer.pad_token_id)

                summaries = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_start_token_id=self.decoder_start_token_id,
                    num_beams=num_generate,
                    num_return_sequences=num_generate,
                    )

                dec = self.tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                decs.append(dec)

            return decs
    
    def get_token_probs(self, contexts, tokens):
        # given context text, and a list of tokens, return the probability of each token
        # context: list of str
        # tokens: list of tokens
        # return: list of probabilities

        probs = []
        for batch in tqdm(list(chunks(contexts, self.batch_size))):
            with torch.no_grad():
                batch = self.tokenizer(batch, return_tensors="pt", truncation=True, padding="max_length").to(self.device)
                input_ids, attention_mask = trim_batch(**batch, pad_token_id=self.tokenizer.pad_token_id)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                # probs = torch.softmax(logits, dim=-1)
                token_ids = self.tokenizer(tokens, add_special_tokens=False, return_tensors="pt").input_ids
                token_probs = torch.softmax(logits[:, 0, token_ids].squeeze(), dim=-1)
                probs.append(token_probs.tolist())
        return probs

all_relations = [
    "AtLocation",
    "CapableOf",
    "Causes",
    "CausesDesire",
    "CreatedBy",
    "DefinedAs",
    "DesireOf",
    "Desires",
    "HasA",
    "HasFirstSubevent",
    "HasLastSubevent",
    "HasPainCharacter",
    "HasPainIntensity",
    "HasPrerequisite",
    "HasProperty",
    "HasSubEvent",
    "HasSubevent",
    "HinderedBy",
    "InheritsFrom",
    "InstanceOf",
    "IsA",
    "LocatedNear",
    "LocationOfAction",
    "MadeOf",
    "MadeUpOf",
    "MotivatedByGoal",
    "NotCapableOf",
    "NotDesires",
    "NotHasA",
    "NotHasProperty",
    "NotIsA",
    "NotMadeOf",
    "ObjectUse",
    "PartOf",
    "ReceivesAction",
    "RelatedTo",
    "SymbolOf",
    "UsedFor",
    "isAfter",
    "isBefore",
    "isFilledBy",
    "oEffect",
    "oReact",
    "oWant",
    "xAttr",
    "xEffect",
    "xIntent",
    "xNeed",
    "xReact",
    "xReason",
    "xWant",
    ]

physical_relations = [
    "AtLocation",
    "CapableOf",
    "CreatedBy",
    "DefinedAs",
    "HasA",
    "HasProperty",
    "InheritsFrom",
    "InstanceOf",
    "IsA",
    "LocatedNear",
    "LocationOfAction",
    "MadeOf",
    "MadeUpOf",
    "NotCapableOf",
    "NotDesires",
    "NotHasA",
    "NotHasProperty",
    "NotIsA",
    "NotMadeOf",
    "PartOf",
    "RelatedTo",
    "SymbolOf",
    "UsedFor",
]

event_relations = [
    "Causes",
    "CausesDesire",
    "HasFirstSubevent",
    "HasLastSubevent",
    "HasPainCharacter",
    "HasPainIntensity",
    "HasPrerequisite",
    "HasSubEvent",
    "HasSubevent",
    "HinderedBy",
    "MotivatedByGoal",
    "ObjectUse",
    "ReceivesAction",
    "isAfter",
    "isBefore",
    "isFilledBy",
    "xAttr",
    "xEffect",
    "xReason",
    "oEffect",
]

affective_relations = [
    "xIntent",
    "xNeed",
    "xReact",
    "xWant",
    "oReact",
    "oWant",
    "DesireOf",
    "Desires",
]

relDecodeConstraint = {
    0: physical_relations,
    1: event_relations,
    2: affective_relations,
}
# USE_CONSTRAINT = False
USE_CONSTRAINT = True


if __name__ == "__main__":
    comet = Comet("ref/comet-atomic-2020/models/comet_atomic2020_bart/comet-atomic_2020_BART")
    comet.model.zero_grad()
    print("model loaded")


    # load datasets
    splits = ["train", "dev", "test"]
    searchDepth = 3
    for s in splits:
        with open(f"data/dataset/{s}Situation.txt", "r") as f:
            situations = []
            for line in f:
                situations.append(line.strip())
        print(f"Loaded {len(situations)} examples from {s} split")

        bestTokens = []
        tokenHistory = [] # for memorizing the token history, to avoid repetition
        for i in range(searchDepth):
            print(f"search depth {i + 1}: ")
            if USE_CONSTRAINT:
                tokens = relDecodeConstraint[i]
            else:
                tokens = all_relations
            print("finding optimal tokens...")
            probs = comet.get_token_probs(situations, tokens)
            probs = [p for l in probs for p in l]
            topTokens = np.array(tokens)[np.argsort(probs, axis=1)[:, -3:][:, ::-1]]
            # topTokens = np.array(tokens)[np.argmax(probs, axis=1)]
            if len(tokenHistory) == 0: tokenHistory = [[] for _ in range(len(topTokens))]
            # avoid adding the tokens that have been used before
            # if a token has been seem, use the second best token, then third
            bestTokens = ["xIntent"]*len(topTokens)
            for j in range(len(topTokens)):
                for k in range(len(topTokens[j])):
                    if topTokens[j][k] not in tokenHistory[j]:
                        bestTokens[j] = topTokens[j][k]
                        break

            for j in range(len(bestTokens)):
                tokenHistory[j].append(bestTokens[j])

            # generate with the top token and add to the situation
            queries = [s + " " + t + " [GEN]" for s, t in zip(situations, bestTokens)]
            print("generating...")
            results = comet.generate(queries, num_generate=1)
            results = [t for l in results for t in l]

            # append the top token and then generated text to the situation
            situations = [s + " " + "[" + t + "]" + r for s, t, r in zip(situations, bestTokens, results)]

        # write to file
        if USE_CONSTRAINT: 
            ver = "relConstraint"
        else:
            ver = "relAll"
        with open(f"data/dataset/{s}Comet_st_{ver}.txt", "w") as f:
            for s in situations:
                f.write(s + "\n")
