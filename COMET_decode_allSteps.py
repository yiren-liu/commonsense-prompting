import re
import json
import os 
import torch
import argparse

import numpy as np
from tqdm import tqdm
from pathlib import Path
from transformers import BartForConditionalGeneration, AutoTokenizer


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
        self.model:BartForConditionalGeneration = BartForConditionalGeneration.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        task = "summarization"
        use_task_specific_params(self.model, task)
        # self.batch_size = 2
        # self.batch_size = 20
        self.batch_size = 40
        if DEBUG: self.batch_size = 2
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

                # avoid none
                bad_words_ids = [[self.tokenizer.encode(bad_word, add_special_tokens=False)[0]] \
                    for bad_word in [" none", "none", "None", "NONE", "help", " help", "sad", " sad"]]
                summaries = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_start_token_id=self.decoder_start_token_id,
                    num_beams=num_generate,
                    num_return_sequences=num_generate,
                    do_sample=True,
                    bad_words_ids=bad_words_ids,
                )

                dec = self.tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                decs.append(np.array(dec).reshape([-1, num_generate]))

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
                token_probs = torch.softmax(logits[:, 0, token_ids].squeeze(dim=-1), dim=-1)
                probs.append(token_probs.tolist())
        return probs

def generate_dialogue_history_files(dataPath):
    splits = ["train", "dev", "test"]
    for split in splits:
        with open(dataPath + f"/{split}DialogueHistory.txt", "w", encoding="utf8") as fw:
            with open(dataPath + f"/{split}WithStrategy_short.tsv", "r", encoding="utf8") as fr:
                # 3 0 0 Hi there, can you help me?  EOS 3 1 1 [Question] I'll do my best. What do you need help with?  EOS 3 0 2 I feel depressed because I had to quit my job and stay home with my kids because of their remote school.  EOS 3 1 3 [Reflection of feelings] I can understand why that would make you feel depressed.  EOS 3 0 4 Do you have any advice on how to feel better?  EOS 3 1 5 [Providing Suggestions] Yes of course. It's good that you are acknowledging your feelings. To improve your mood you could practice hobbies or other things you enjoy doing. 
                for line in fr:
                    lines = line.strip().split(" EOS ")
                    newLine = ""
                    for i in range(len(lines)):
                        # tokens = lines[i].split(" ")
                        if re.findall(r"^\ ?\[.*\]", lines[i][6:]):
                            role = "PersonY"
                            tokens = lines[i].split("] ")
                            text = " ".join(tokens[1:])
                        else:
                            role = "PersonX"
                            tokens = lines[i].split(" ")
                            text = " ".join(tokens[3:])
                        newLine += f"{role}: {text} EOS "
                    fw.write(newLine.strip() + '\n')


comet_only_relations = [
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

# configs 
# USE_CONSTRAINT = False
USE_CONSTRAINT = True

USE_DIALOGUE_HISTORY = True

# USE_LAST_UTTERANCE = True
USE_LAST_UTTERANCE = False

AVOID_REPETITION = True # when decoding, avoid repeating the same entailment

DECODE_ALL = True # decode all relations independently, and dump to json

COMET_REL_ONLY = True # only decode relations in comet_only_relations
if COMET_REL_ONLY: 
    # take the intersection of comet_only_relations and other sets
    all_relations = list(set(all_relations) & set(comet_only_relations))
    physical_relations = list(set(physical_relations) & set(comet_only_relations))
    event_relations = list(set(event_relations) & set(comet_only_relations))
    affective_relations = list(set(affective_relations) & set(comet_only_relations))

DEBUG = False
# DEBUG = True

ONLY_LAST = True # only decode the last utterance


# TODO: add dialogue summarization



if __name__ == "__main__":
    relDecodeConstraint = {
        0: physical_relations,
        1: event_relations,
        2: affective_relations,
    }

    dataPath = "data/dataset"
    if USE_DIALOGUE_HISTORY:
        # check if the data file exists
        if not (
            os.path.exists(f"{dataPath}/trainDialogueHistory.txt")
            and os.path.exists(f"{dataPath}/devDialogueHistory.txt")
            and os.path.exists(f"{dataPath}/testDialogueHistory.txt")
        ):
            print("dialogue distory file not found, generating...")
            generate_dialogue_history_files(dataPath)


    comet = Comet("ref/comet-atomic-2020/models/comet_atomic2020_bart/comet-atomic_2020_BART")
    comet.model.zero_grad()
    print("model loaded")        

    # load datasets
    splits = ["train", "dev", "test"]
    searchDepth = 3
    if COMET_REL_ONLY:
        relDecodeConstraint = {
            0: event_relations,
            1: affective_relations,
        }
        searchDepth = 2
    
    for s in splits:
        if USE_DIALOGUE_HISTORY:
            dataName = "DialogueHistory"
        else:    
            dataName = "Situation"
        with open(f"{dataPath}/{s}{dataName}.txt", "r", encoding='utf8') as f:
            situations = []
            cnt = 0
            for line in f:
                if DEBUG and cnt > 3: break
                # situations.append(line.strip())
                uttrs = [uttr.strip() for uttr in line.strip().split(' EOS') if uttr]
                situations.append(uttrs)
                # raise NotImplementedError
                cnt += 1
        print(f"Loaded {len(situations)} examples from {s} split")

        if DECODE_ALL:
            # geberate using all relations independently
            queries = []
            situRel = [] # [(s, r), ...]
            for situ in situations:
                if ONLY_LAST: situ = [situ[-1]]
                for i in range(len(situ)):
                    for r in comet_only_relations:
                        # take conversations upto i
                        q = ' EOS '.join(situ[:i+1])
                        queries.append(q + " " + r + " [GEN]")
                        situRel.append((' EOS '.join(situ), i, r))
            # generation with comet model
            print("generating...")
            results = comet.generate(queries, num_generate=5)

            # unpack the results
            temp = {}
            situIdx = 0
            for batch in results:
                for topk in batch:
                    situ, i, r = situRel[situIdx]
                    if situ not in temp:
                        temp[situ] = {}
                    if i not in temp[situ]:
                        temp[situ][i] = {}
                    temp[situ][i][r] = topk.tolist()
                    situIdx += 1
            results = temp

            # convert and dump to jsonl
            tag = "allSteps"
            if ONLY_LAST: tag = "lastStep"
            with open(f"{dataPath}/{s}CometOnly_{dataName}_ind_{tag}.jsonl", "w", encoding='utf8') as f:
                for s in results:
                    # for r in results[s]:
                    f.write(
                            json.dumps(
                                {
                                    "situation": s,
                                    "entailments": results[s],
                                }
                            )
                            + "\n"
                        )
        else:
            raise NotImplementedError
            
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
                results = comet.generate(queries, num_generate=3)
                # avoid generated none

                # results = [t for l in results for t in l]
                # results = [t for l in results for topk in l for t in topk if "none" not in t]
                temp = []
                situIdx = 0
                for batch in results:
                    for topk in batch:
                        for t in topk:
                            if "none" not in t and (not AVOID_REPETITION or t not in situations[situIdx]):
                                temp.append(t)
                                situIdx += 1
                                break
                results = temp
                

                # append the top token and then generated text to the situation
                situations = [s + " " + "[" + t + "]" + r for s, t, r in zip(situations, bestTokens, results)]

            # write to file
            if USE_CONSTRAINT: 
                ver = "relConstraint"
            else:
                ver = "relAll"
            if USE_DIALOGUE_HISTORY:
                dataName = "dialog"
                if USE_LAST_UTTERANCE:
                    dataName += "Last"
            else:    
                dataName = "st"
            if COMET_REL_ONLY:
                relSet = "CometOnly"
            else:
                relSet = "All"
            with open(f"{dataPath}/{s}Comet_{dataName}_{ver}_{relSet}.txt", "w", encoding='utf8') as f:
                for s in situations:
                    f.write(s + "\n")
