import os
import pickle
import logging
from unittest import skip
import json

import pandas as pd
import numpy as np
from tokenizers import Tokenizer
import torch

from tqdm import tqdm

from sklearn.model_selection import train_test_split

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import PreTrainedTokenizer

# Configs
logger = logging.getLogger(__name__)
STRATEGYLIST = [
    "[Question]", "[Reflection of feelings]", "[Information]", "[Restatement or Paraphrasing]",
    "[Others]", "[Self-disclosure]", "[Affirmation and Reassurance]", "[Providing Suggestions]",
    "[None]"
]



# https://github.com/morecry/MISC
class ESDDatasetBlenderbot(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, df, comet, comet_st, block_size=512, evaluate=False, strategy=True, test=False):
        block_size = block_size - (tokenizer.model_max_length - tokenizer.max_len_single_sentence)
        self.tokenizer = tokenizer
        directory = os.path.join(args.data_cache_dir, args.data_path.split('/')[-1])
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        if evaluate:
            if not test:
                cached_features_file = os.path.join(
                    directory, 'val_' + args.model_type + "_cached_lm_" + str(block_size)
                )
            else:
                cached_features_file = os.path.join(
                    directory, 'test_' + args.model_type + "_cached_lm_" + str(block_size)
                )
        else:
            cached_features_file = os.path.join(
                directory, 'trn_' + args.model_type + "_cached_lm_" + str(block_size)
            )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.features = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)
            print(len(df) , len(comet), len(comet_st))
            assert len(df) == len(comet) == len(comet_st)
            self.features = []
            print("loading data from files...")
            for idx, (row, comet_row, comet_st_row) in enumerate(tqdm(zip(df[:-1], comet[:-1], comet_st[:-1]), total=len(df[:-1]))):
                conv = self.construct_conv_ESD_blenderbot(idx, row, comet_row, comet_st_row, tokenizer, cls=False, strategy=strategy ,evaluate=evaluate)
                if len(conv.input_ids) >= block_size:
                    conv.input_ids = conv.input_ids[-block_size:]
                    conv.input_ids[0] = tokenizer.encode(tokenizer.cls_token)[0]
                else:
                    conv.input_ids = tokenizer.encode(tokenizer.cls_token) + conv.input_ids
                self.features.append(conv)

            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.features, handle, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Finished~")
    
    
    def construct_conv_ESD_blenderbot(self, idx, row, comet_row, comet_st_row, tokenizer, eos = True, pad=True, cls=False, evaluate=False, strategy=True, generation=False):
        #  process input text
        inputs, roles, turns, strategy_labels, _ = self._get_inputs_from_text_blenderbot("EOS".join(row.split("EOS")[:-1]), tokenizer, strategy=strategy)
        # process output (decoder input) text
        d_inputs, d_roles, d_turns, d_strategy_labels, emotion = self._get_inputs_from_text_blenderbot(row.split("EOS")[-1], tokenizer, strategy=strategy)

        
        # print("EOS".join(row.split("EOS")[:-1]))
        # print(row.split("EOS")[-1])
        # raise Exception("debug")

        # make feature for input text
        feature = self._make_feature_blenderbot(idx, inputs, roles, turns, tokenizer.eos_token_id, pad=pad, strategy_labels=strategy_labels, evaluate=evaluate, str_embd=True, generation=generation)
        # make feature for output (decoder input) text
        d_feature = self._make_feature_blenderbot(idx, d_inputs, d_roles, d_turns, tokenizer.eos_token_id, pad=pad, strategy_labels=d_strategy_labels, evaluate=evaluate, str_embd=True, generation=generation)
        comet_ids, comet_mask = self._get_comet_input_blenderbot(comet_row, tokenizer)
        comet_st_ids, comet_st_mask = self._get_comet_input_blenderbot(comet_st_row, tokenizer, max_num_attr=20)
        feature = InputFeatures_blender(feature, d_feature, comet_ids, comet_mask, emotion, comet_st_ids, comet_st_mask)
        return feature



    def _get_comet_input_blenderbot(self, comet_row, tokenizer, max_num_attr=30, max_len_attr=10):
        attrs = comet_row.split('__EOS__')[:-1]
        comet_ids = []
        comet_mask = []
        for ids, attr in enumerate(attrs):
            if ids == max_num_attr:
                break
            comet_attr_ids = tokenizer.encode(attr)
            if len(comet_attr_ids) < max_len_attr:
                comet_attr_ids += [tokenizer.pad_token_id]*(max_len_attr - len(comet_attr_ids)) 
            else:
                comet_attr_ids = comet_attr_ids[:max_len_attr]
            comet_ids.append(comet_attr_ids)
            comet_mask.append(1)

        if len(comet_ids) < max_num_attr:
            comet_ids += ([[tokenizer.pad_token_id]*max_len_attr]) * (max_num_attr - len(comet_ids))
            comet_mask += [0] * (max_num_attr - len(comet_mask))
        # print(attrs) 
        # print(comet_ids)
        # print(comet_mask)
        # print(error)
        
        assert len(comet_ids) == max_num_attr
        assert len(comet_mask) == max_num_attr
        return comet_ids, comet_mask

    def _get_inputs_from_text_blenderbot(self, text, tokenizer, strategy=True, cls = False):
        srcs = text.strip()
        inputs = []
        roles = []
        turns = []
        strategy_labels=[]
        srcs = srcs.split(" EOS")
        emotion = None
        for idx, src in enumerate(srcs):

            if src =="":
                continue
            src_emo, src_role, src_turn, src = self._norm_text(src)
            if emotion is None:
                emotion = src_emo

            context_id = tokenizer.encode(src)

            if not strategy:
                context_id = [i  for i in context_id if i< 50257+4687]
            elif cls:
                context_id = tokenizer.cls + [i for i in context_id if i< 50257+4687]
            else:
                pass

            if src_role==1:
                try:
                    label = "["+src.split("[")[1].split("]")[0]+"]"
                except Exception as e:
                    strategy_labels.append(8)
                else:
                    strategy_labels.append(tokenizer.encode([label])[0] - 50257-4687)
            else:
                strategy_labels.append(8)

            inputs.append(context_id)
            roles.append(src_role)
            turns.append(src_turn)

        return inputs, roles, turns, strategy_labels, emotion


    def _make_feature_blenderbot(self, id_, sents, rls, ts, eos, pad=False, block_size=512, strategy_labels=None, evaluate=False, str_embd=False, generation=False):
        # we did't use role label and turn number in modeling as they did't carry significant improvement. However, codes still remain here.
        if len(sents) == 0:
            return InputFeatures_train([], [], [], [], [],
                                [], [] , [], [])
        input_ids = [i for s in sents for i in s+[eos]]

        input_ids = input_ids
        lm_labels = []
        token_type_ids = []
        roles = []
        strategy_ids = []

        for i, s in enumerate(sents):
            token_type_ids += [ts[i]] * (len(s) + 1)
            flag_str = -1
            if str_embd: #use for strategy embed but currently we treat strategy as token
                strategy_ids += [strategy_labels[-1]] * (len(s) + 1)
            else:
                strategy_ids += [8] * (len(s) + 1)
            if i < len(sents) - 1:
                lm_labels += [-100] * (len(s) + 1)
                roles += [rls[i]] * (len(s) + 1)
            else:
                lm_labels += (  s + [eos])
                roles += [rls[i]] * (len(s) + 1)

        i = len(lm_labels) - 1
        if len(input_ids) == 1:
            print(input_ids, lm_labels, token_type_ids, roles)
        while i >= 0:
            if lm_labels[i] != -100:
                break
            i -= 1
        input_ids = input_ids[:i+1]
        lm_labels = lm_labels[:i+1]
        token_type_ids = token_type_ids[:i+1]
        roles = roles[:i+1]
        if not str_embd:
            strategy_ids = [8]*len(input_ids) # strategy is not used
        else:
            strategy_ids = strategy_ids[:i+1]
        if len(input_ids) == 1:
            print(input_ids, lm_labels, token_type_ids, roles)


        assert (len(input_ids) == len(token_type_ids)
                == len(lm_labels) == len(roles) == len(strategy_ids))
        # cut according to block size
        if len(input_ids) > block_size:
            cut_index = input_ids.index(eos,-512) + 1
            input_ids = input_ids[cut_index: ]

            token_type_ids = token_type_ids[cut_index: ]
            lm_labels = lm_labels[cut_index: ]
            roles = roles[cut_index: ]
            strategy_ids = strategy_ids[cut_index: ]
        # pad to multiples of 8
        if pad:
            while len(input_ids) % 8 != 0:
                input_ids.append(0)
                token_type_ids.append(0)
                lm_labels.append(-100)
                roles.append(0)
                strategy_ids.append(8)
            assert len(input_ids) % 8 == 0
        position_ids = list(range(len(input_ids)))
        assert (len(input_ids) == len(position_ids) == len(token_type_ids)
                == len(lm_labels) == len(roles) == len(strategy_ids))
        if len(input_ids) == 0:
            import pdb
            pdb.set_trace()
        elif len(input_ids) == 1:
            print(input_ids, lm_labels, token_type_ids, roles)
        if True:
            # if it is for generation, the last sentence of context is the last sentence
            cls_position = len(input_ids)-1-input_ids[::-1].index(eos)
        else:
            # if not, the last sentence of context is the second last sentence
            cls_position = len(input_ids)-1-input_ids[::-1].index(eos,input_ids[::-1].index(eos)+1)
        if evaluate and strategy_labels[-1]!=8:
            try:
                lm_labels[lm_labels.index(strategy_labels[-1]+50257+4687)] = -100
            except Exception:
                pass

        feature = InputFeatures_train(id_, input_ids, position_ids, token_type_ids, roles,
                                lm_labels, cls_position , strategy_labels[-1], strategy_ids)
        return feature

    def _norm_text(self, text):
        emo, r, t, *toks = text.strip().split()
        try:
            emo = int(emo)
            r = int(r)
            t = int(t)
            toks = ' '.join(toks[:len(toks)])
        except Exception as e:
            raise e
        return emo, r, t, toks



    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        return self.features[item]

    @staticmethod
    def collate(features):
        input_ids = pad_sequence([torch.tensor(f.input_ids, dtype=torch.long)
                                  for f in features],
                                 batch_first=True, padding_value=0)

        position_ids = pad_sequence([torch.tensor(f.position_ids,
                                                  dtype=torch.long)
                                     for f in features],
                                    batch_first=True, padding_value=0)
        token_type_ids = pad_sequence([torch.tensor(f.token_type_ids,
                                                    dtype=torch.long)
                                       for f in features],
                                      batch_first=True, padding_value=0)
        role_ids = pad_sequence([torch.tensor(f.role_ids, 
                                              dtype=torch.long)
                                      for f in features],
                                     batch_first=True, padding_value=0)
        labels = pad_sequence([torch.tensor(f.lm_labels, dtype=torch.long)
                               for f in features],
                              batch_first=True, padding_value=-100)
        
        cls_positions = torch.tensor([f.cls_position for f in features], dtype=torch.long)
        
        cls_labels = torch.tensor([f.cls_label for f in features], dtype=torch.long)
        
        strategy_ids = pad_sequence([torch.tensor(f.strategy_ids, dtype=torch.long)
                               for f in features],
                              batch_first=True, padding_value=8)

        decoder_input_ids = pad_sequence([torch.tensor(f.decoder_input_ids, dtype=torch.long)
                                  for f in features],
                                 batch_first=True, padding_value=0)
        decoder_position_ids = pad_sequence([torch.tensor(f.decoder_position_ids,
                                                  dtype=torch.long)
                                     for f in features],
                                    batch_first=True, padding_value=0)
        decoder_token_type_ids = pad_sequence([torch.tensor(f.decoder_token_type_ids,
                                                    dtype=torch.long)
                                       for f in features],
                                      batch_first=True, padding_value=0)
        decoder_role_ids = pad_sequence([torch.tensor(f.decoder_role_ids,
                                              dtype=torch.long)
                                      for f in features],
                                     batch_first=True, padding_value=0)
        decoder_labels = pad_sequence([torch.tensor(f.decoder_lm_labels, dtype=torch.long)
                               for f in features],
                              batch_first=True, padding_value=-100)

        decoder_cls_positions = torch.tensor([f.decoder_cls_position for f in features], dtype=torch.long)

        decoder_cls_labels = torch.tensor([f.decoder_cls_label for f in features], dtype=torch.long)

        decoder_strategy_ids = pad_sequence([torch.tensor(f.decoder_strategy_ids, dtype=torch.long)
                               for f in features],
                              batch_first=True, padding_value=8)
        # print([f.comet_ids for f in features])
        # print([f.comet_mask for f in features])
        comet_ids = torch.tensor([f.comet_ids for f in features], dtype=torch.long)
        comet_mask = torch.tensor([f.comet_mask for f in features], dtype=torch.long)
        emotion = torch.tensor([f.emotion for f in features], dtype=torch.long)
        comet_st_ids = torch.tensor([f.comet_st_ids for f in features], dtype=torch.long)
        comet_st_mask = torch.tensor([f.comet_st_mask for f in features], dtype=torch.long)

        return (input_ids, position_ids, token_type_ids, role_ids, labels, cls_positions, cls_labels, strategy_ids, decoder_input_ids, decoder_position_ids, decoder_token_type_ids, decoder_role_ids, decoder_labels, decoder_cls_positions, decoder_cls_labels, decoder_strategy_ids, comet_ids, comet_mask, emotion, comet_st_ids, comet_st_mask)


class ESDDatasetBartCOMET2020(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, df, comet, comet_st, st, comet_by_step=None, block_size=512, evaluate=False, strategy=True, test=False, add_situ=True):
        block_size = block_size - (tokenizer.model_max_length - tokenizer.max_len_single_sentence)
        self.tokenizer = tokenizer
        self.strategyNull = self.tokenizer.encode('[None]', add_special_tokens=False)[0]
        self.strategyIds = [self.tokenizer.encode(t, add_special_tokens=False)[0] for t in STRATEGYLIST]

        directory = os.path.join(args.data_cache_dir, args.data_path.split('/')[-1])
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        if evaluate:
            if not test:
                cached_features_file = os.path.join(
                    directory, 'val_' + args.model_type + "_cached_lm_" + str(block_size)
                )
            else:
                cached_features_file = os.path.join(
                    directory, 'test_' + args.model_type + "_cached_lm_" + str(block_size)
                )
        else:
            cached_features_file = os.path.join(
                directory, 'trn_' + args.model_type + "_cached_lm_" + str(block_size)
            )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.features = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)
            print(len(df) , len(comet), len(comet_st))
            assert len(df) == len(comet) == len(comet_st)
            self.features = []
            print("loading data from files...")
            if comet_by_step and args.append_comet_to_input:
                for idx, (row, comet_row, comet_st_row, st_row, comet_by_step) in enumerate(tqdm(zip(df[:-1], comet[:-1], comet_st[:-1], st[:-1], comet_by_step[:-1]), total=len(df[:-1]))):
                    conv = self.construct_conv_ESD(idx, row, comet_row, comet_st_row, st_row, tokenizer, comet_by_step=comet_by_step, cls=False, strategy=strategy ,evaluate=evaluate, add_situ=add_situ)
                    if len(conv.input_ids) >= block_size:
                        conv.input_ids = conv.input_ids[-block_size:]
                        # conv.input_ids[0] = tokenizer.encode(tokenizer.cls_token)[0]
                    # else:
                    #     conv.input_ids = tokenizer.encode(tokenizer.cls_token) + conv.input_ids
                    self.features.append(conv)
            else:
                for idx, (row, comet_row, comet_st_row, st_row) in enumerate(tqdm(zip(df[:-1], comet[:-1], comet_st[:-1], st[:-1]), total=len(df[:-1]))):
                    conv = self.construct_conv_ESD(idx, row, comet_row, comet_st_row, st_row, tokenizer, cls=False, strategy=strategy ,evaluate=evaluate, add_situ=add_situ)
                    if len(conv.input_ids) >= block_size:
                        conv.input_ids = conv.input_ids[-block_size:]
                        # conv.input_ids[0] = tokenizer.encode(tokenizer.cls_token)[0]
                    # else:
                    #     conv.input_ids = tokenizer.encode(tokenizer.cls_token) + conv.input_ids
                    self.features.append(conv)

            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.features, handle, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Finished~")
    
    
    def construct_conv_ESD(self, idx, row, comet_row, comet_st_row, st_row, tokenizer, comet_by_step=None, eos = True, pad=True, cls=False, evaluate=False, strategy=True, generation=False, add_situ=True):
        #  process input text
        # inputs, roles, turns, strategy_labels, _ = self._get_inputs_from_text("EOS".join(row.split("EOS")[:-1]), tokenizer, strategy=strategy, add_gen=True)
        inputs, roles, turns, strategy_labels, _ = self._get_inputs_from_text("EOS".join(row.split("EOS")[:-1]), st_row, tokenizer, comet_by_step, strategy=strategy, add_situ=add_situ)
        # process output (decoder input) text
        d_inputs, d_roles, d_turns, d_strategy_labels, emotion = self._get_inputs_from_text(row.split("EOS")[-1], st_row, tokenizer, strategy=True, add_situ=False, sos=False)
        
        
        # print("EOS".join(row.split("EOS")[:-1]))
        # print(row.split("EOS")[-1])
        # raise Exception("debug")

        # make feature for input text
        feature = self._make_feature(idx, inputs, roles, turns, tokenizer.eos_token_id, pad=pad, strategy_labels=strategy_labels, evaluate=evaluate, str_embd=True, generation=generation)
        # make feature for output (decoder input) text
        d_feature = self._make_feature(idx, d_inputs, d_roles, d_turns, tokenizer.eos_token_id, pad=pad, strategy_labels=d_strategy_labels, evaluate=evaluate, str_embd=True, generation=generation)


        comet_ids, comet_mask = self._get_comet_input(comet_row, tokenizer)
        comet_st_ids, comet_st_mask = self._get_comet_input(comet_st_row, tokenizer, max_num_attr=20)
        
        if comet_by_step:
            comet_by_step_ids, comet_by_step_mask = self._get_comet_by_step_input(comet_by_step, tokenizer, max_num_attr=20)
        else:
            comet_by_step_ids, comet_by_step_mask = None, None
        
        
        # raise Exception("debug")
        
        feature = InputFeatures_blender(feature, d_feature, comet_ids, comet_mask, emotion, comet_st_ids, comet_st_mask, comet_by_step_ids, comet_by_step_mask)
        return feature

    

    def _get_comet_by_step_input(self, comet_row, tokenizer, max_num_attr=30, max_len_attr=20):
        attrs = json.loads(comet_row)
        last_attrs = list(attrs['entailments'].values())[-1]
        
        attrs = []
        for r, t in last_attrs.items():
            attrs.append(f"[{r}] {t[0]}")

        max_num_attr = min(max_num_attr, len(attrs))

        comet_ids = []
        comet_mask = []
        for ids, attr in enumerate(attrs):
            if ids == max_num_attr:
                break
            comet_attr_ids = tokenizer.encode(attr)
            if len(comet_attr_ids) < max_len_attr:
                comet_attr_ids += [tokenizer.pad_token_id]*(max_len_attr - len(comet_attr_ids)) 
            else:
                comet_attr_ids = comet_attr_ids[:max_len_attr]
            comet_ids.append(comet_attr_ids)
            comet_mask.append(1)

        if len(comet_ids) < max_num_attr:
            comet_ids += ([[tokenizer.pad_token_id]*max_len_attr]) * (max_num_attr - len(comet_ids))
            comet_mask += [0] * (max_num_attr - len(comet_mask))
        # print(attrs) 
        # print(comet_ids)
        # print(comet_mask)
        # print(error)
        
        assert len(comet_ids) == max_num_attr
        assert len(comet_mask) == max_num_attr
        return comet_ids, comet_mask

    def _get_comet_input(self, comet_row, tokenizer, max_num_attr=30, max_len_attr=10):
        attrs = comet_row.split('__EOS__')[:-1]
        comet_ids = []
        comet_mask = []
        for ids, attr in enumerate(attrs):
            if ids == max_num_attr:
                break
            comet_attr_ids = tokenizer.encode(attr)
            if len(comet_attr_ids) < max_len_attr:
                comet_attr_ids += [tokenizer.pad_token_id]*(max_len_attr - len(comet_attr_ids)) 
            else:
                comet_attr_ids = comet_attr_ids[:max_len_attr]
            comet_ids.append(comet_attr_ids)
            comet_mask.append(1)

        if len(comet_ids) < max_num_attr:
            comet_ids += ([[tokenizer.pad_token_id]*max_len_attr]) * (max_num_attr - len(comet_ids))
            comet_mask += [0] * (max_num_attr - len(comet_mask))
        # print(attrs) 
        # print(comet_ids)
        # print(comet_mask)
        # print(error)
        
        assert len(comet_ids) == max_num_attr
        assert len(comet_mask) == max_num_attr
        return comet_ids, comet_mask

    def _get_inputs_from_text(self, text, st_row, tokenizer, comet_by_step=None, strategy=False, cls = False, add_gen=False, add_situ=False, sos=True):
        srcs = text.strip()
        inputs = []
        roles = []
        turns = []
        strategy_labels=[]

        # adding tokens before encoding
        # srcs += " [GEN]"
        # srcs += " Response:"
        # if add_gen: srcs += " [GEN]"
        if add_gen: srcs += " Response:"
        if add_situ: srcs = "0 0 0 Context:" + st_row.strip() + " EOS" + srcs
        if comet_by_step:
            attrs = json.loads(comet_by_step)
            last_attrs = list(attrs['entailments'].values())[-1]
            attrs = []
            for r, t in last_attrs.items():
                attrs.append(f"[{r}] {t[0]}")
            srcs += " EOS " + "0 0 0 " + " ".join(attrs[:3])


        srcs = srcs.split(" EOS")
        emotion = None
        for idx, src in enumerate(srcs):

            if src =="":
                continue
            src_emo, src_role, src_turn, src = self._norm_text(src)
            if emotion is None:
                emotion = src_emo

            context_id = tokenizer.encode(src)
            if not sos: context_id = context_id[1:]

            if not strategy:
                context_id = [i for i in context_id if i not in self.strategyIds]
            elif cls:
                context_id = tokenizer.cls + [i for i in context_id if i not in self.strategyIds]
            else:
                pass

            if src_role==1:
                try:
                    label = "["+src.split("[")[1].split("]")[0]+"]"
                except Exception as e:
                    strategy_labels.append(self.strategyNull)
                else:
                    strategy_labels.append(tokenizer.encode(label)[1])
                    # print(tokenizer.decode(self.strategyNull))
                    # raise Exception("debug")
            else:
                strategy_labels.append(self.strategyNull)

            inputs.append(context_id)
            roles.append(src_role)
            turns.append(src_turn)

        return inputs, roles, turns, strategy_labels, emotion


    def _make_feature(self, id_, sents, rls, ts, eos, pad=False, block_size=512, strategy_labels=None, evaluate=False, str_embd=False, generation=False):
        # we did't use role label and turn number in modeling as they did't carry significant improvement. However, codes still remain here.
        if len(sents) == 0:
            return InputFeatures_train([], [], [], [], [],
                                [], [] , [], [])
        input_ids = [i for s in sents for i in s]

        input_ids = input_ids
        lm_labels = []
        token_type_ids = []
        roles = []
        strategy_ids = []

        for i, s in enumerate(sents):
            token_type_ids += [ts[i]] * (len(s))
            flag_str = -1
            if str_embd: #use for strategy embed but currently we treat strategy as token
                strategy_ids += [strategy_labels[-1]] * (len(s))
            else:
                strategy_ids += [self.strategyNull] * (len(s))
            if i < len(sents) - 1:
                lm_labels += [-100] * (len(s))
                roles += [rls[i]] * (len(s))
            else:
                lm_labels += (  s )
                roles += [rls[i]] * (len(s))

        i = len(lm_labels) - 1
        if len(input_ids) == 1:
            print(input_ids, lm_labels, token_type_ids, roles)
        while i >= 0:
            if lm_labels[i] != -100:
                break
            i -= 1
        input_ids = input_ids[:i+1]
        lm_labels = lm_labels[:i+1]
        token_type_ids = token_type_ids[:i+1]
        roles = roles[:i+1]
        if not str_embd:
            strategy_ids = [self.strategyNull]*len(input_ids) # strategy is not used
        else:
            strategy_ids = strategy_ids[:i+1]
        if len(input_ids) == 1:
            print(input_ids, lm_labels, token_type_ids, roles)


        assert (len(input_ids) == len(token_type_ids)
                == len(lm_labels) == len(roles) == len(strategy_ids))
        # cut according to block size
        if len(input_ids) > block_size:
            cut_index = input_ids.index(eos,-512) + 1
            input_ids = input_ids[cut_index: ]

            token_type_ids = token_type_ids[cut_index: ]
            lm_labels = lm_labels[cut_index: ]
            roles = roles[cut_index: ]
            strategy_ids = strategy_ids[cut_index: ]
        # pad to multiples of 8
        if pad:
            while len(input_ids) % 8 != 0:
                input_ids.append(1)
                token_type_ids.append(1)
                lm_labels.append(-100)
                roles.append(1)
                strategy_ids.append(self.strategyNull)
            assert len(input_ids) % 8 == 0
        position_ids = list(range(len(input_ids)))
        assert (len(input_ids) == len(position_ids) == len(token_type_ids)
                == len(lm_labels) == len(roles) == len(strategy_ids))
        if len(input_ids) == 0:
            import pdb
            pdb.set_trace()
        elif len(input_ids) == 1:
            print(input_ids, lm_labels, token_type_ids, roles)
        if True:
            # if it is for generation, the last sentence of context is the last sentence
            cls_position = len(input_ids)-1-input_ids[::-1].index(eos)
        else:
            # if not, the last sentence of context is the second last sentence
            cls_position = len(input_ids)-1-input_ids[::-1].index(eos,input_ids[::-1].index(eos)+1)
        
        # if evaluate and strategy_labels[-1]!=self.strategyNull:
        #     try:
        #         lm_labels[lm_labels.index(strategy_labels[-1]+50257+4687)] = -100
        #     except Exception:
        #         pass

        feature = InputFeatures_train(id_, input_ids, position_ids, token_type_ids, roles,
                                lm_labels, cls_position , strategy_labels[-1], strategy_ids)
        return feature

    def _norm_text(self, text):
        emo, r, t, *toks = text.strip().split()
        try:
            emo = int(emo)
            r = int(r)
            t = int(t)
            toks = ' '.join(toks[:len(toks)])
        except Exception as e:
            raise e
        return emo, r, t, toks



    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        return self.features[item]

    @staticmethod
    def collate(features):
        input_ids = pad_sequence([torch.tensor(f.input_ids, dtype=torch.long)
                                  for f in features],
                                 batch_first=True, padding_value=1)

        position_ids = pad_sequence([torch.tensor(f.position_ids,
                                                  dtype=torch.long)
                                     for f in features],
                                    batch_first=True, padding_value=1)
        token_type_ids = pad_sequence([torch.tensor(f.token_type_ids,
                                                    dtype=torch.long)
                                       for f in features],
                                      batch_first=True, padding_value=1)
        role_ids = pad_sequence([torch.tensor(f.role_ids, 
                                              dtype=torch.long)
                                      for f in features],
                                     batch_first=True, padding_value=1)
        labels = pad_sequence([torch.tensor(f.lm_labels, dtype=torch.long)
                               for f in features],
                              batch_first=True, padding_value=-100)
        
        cls_positions = torch.tensor([f.cls_position for f in features], dtype=torch.long)
        
        cls_labels = torch.tensor([f.cls_label for f in features], dtype=torch.long)
        
        strategy_ids = pad_sequence([torch.tensor(f.strategy_ids, dtype=torch.long)
                               for f in features],
                              batch_first=True, padding_value=1)

        decoder_input_ids = pad_sequence([torch.tensor(f.decoder_input_ids, dtype=torch.long)
                                  for f in features],
                                 batch_first=True, padding_value=1)
        decoder_position_ids = pad_sequence([torch.tensor(f.decoder_position_ids,
                                                  dtype=torch.long)
                                     for f in features],
                                    batch_first=True, padding_value=1)
        decoder_token_type_ids = pad_sequence([torch.tensor(f.decoder_token_type_ids,
                                                    dtype=torch.long)
                                       for f in features],
                                      batch_first=True, padding_value=1)
        decoder_role_ids = pad_sequence([torch.tensor(f.decoder_role_ids,
                                              dtype=torch.long)
                                      for f in features],
                                     batch_first=True, padding_value=1)
        decoder_labels = pad_sequence([torch.tensor(f.decoder_lm_labels, dtype=torch.long)
                               for f in features],
                              batch_first=True, padding_value=-100)

        decoder_cls_positions = torch.tensor([f.decoder_cls_position for f in features], dtype=torch.long)

        decoder_cls_labels = torch.tensor([f.decoder_cls_label for f in features], dtype=torch.long)

        decoder_strategy_ids = pad_sequence([torch.tensor(f.decoder_strategy_ids, dtype=torch.long)
                               for f in features],
                              batch_first=True, padding_value=1)
        # print([f.comet_ids for f in features])
        # print([f.comet_mask for f in features])
        comet_ids = torch.tensor([f.comet_ids for f in features], dtype=torch.long)
        comet_mask = torch.tensor([f.comet_mask for f in features], dtype=torch.long)
        emotion = torch.tensor([f.emotion for f in features], dtype=torch.long)
        comet_st_ids = torch.tensor([f.comet_st_ids for f in features], dtype=torch.long)
        comet_st_mask = torch.tensor([f.comet_st_mask for f in features], dtype=torch.long)

        if features[0].comet_by_step_ids:
            comet_by_step_ids = torch.tensor([f.comet_by_step_ids for f in features], dtype=torch.long)
            comet_by_step_mask = torch.tensor([f.comet_by_step_mask for f in features], dtype=torch.long)
        else:
            comet_by_step_ids = None
            comet_by_step_mask = None

        return (input_ids, position_ids, token_type_ids, role_ids, labels, cls_positions, cls_labels, strategy_ids, decoder_input_ids, decoder_position_ids, decoder_token_type_ids, decoder_role_ids, decoder_labels, decoder_cls_positions, decoder_cls_labels, decoder_strategy_ids, comet_ids, comet_mask, emotion, comet_st_ids, comet_st_mask, comet_by_step_ids, comet_by_step_mask)






class ESDDatasetGPT2COMET2020(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, df, comet, comet_st, st, block_size=512, evaluate=False, strategy=True, test=False, add_situ=True):
        block_size = block_size - (tokenizer.model_max_length - tokenizer.max_len_single_sentence)
        self.tokenizer = tokenizer
        self.strategyNull = self.tokenizer.encode('[None]', add_special_tokens=False)[0]
        self.strategyIds = [self.tokenizer.encode(t, add_special_tokens=False)[0] for t in STRATEGYLIST]

        directory = os.path.join(args.data_cache_dir, args.data_path.split('/')[-1])
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        if evaluate:
            if not test:
                cached_features_file = os.path.join(
                    directory, 'val_' + args.model_type + "_cached_lm_" + str(block_size)
                )
            else:
                cached_features_file = os.path.join(
                    directory, 'test_' + args.model_type + "_cached_lm_" + str(block_size)
                )
        else:
            cached_features_file = os.path.join(
                directory, 'trn_' + args.model_type + "_cached_lm_" + str(block_size)
            )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.features = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)
            print(len(df) , len(comet), len(comet_st))
            assert len(df) == len(comet) == len(comet_st)
            self.features = []
            print("loading data from files...")
            for idx, (row, comet_row, comet_st_row, st_row) in enumerate(tqdm(zip(df[:-1], comet[:-1], comet_st[:-1], st[:-1]), total=len(df[:-1]))):
                conv = self.construct_conv_ESD(idx, row, comet_row, comet_st_row, st_row, tokenizer, cls=False, strategy=strategy ,evaluate=evaluate, add_situ=add_situ)
                if len(conv.input_ids) >= block_size:
                    conv.input_ids = conv.input_ids[-block_size:]
                    # conv.input_ids[0] = tokenizer.encode(tokenizer.cls_token)[0]
                # else:
                #     conv.input_ids = tokenizer.encode(tokenizer.cls_token) + conv.input_ids
                self.features.append(conv)

            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.features, handle, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Finished~")
    
    
    def construct_conv_ESD(self, idx, row, comet_row, comet_st_row, st_row, tokenizer, eos = True, pad=True, cls=False, evaluate=False, strategy=True, generation=False, add_situ=True):
        #  process input text
        # inputs, roles, turns, strategy_labels, _ = self._get_inputs_from_text("EOS".join(row.split("EOS")[:-1]), tokenizer, strategy=strategy, add_gen=True)
        inputs, roles, turns, strategy_labels, _ = self._get_inputs_from_text("EOS".join(row.split("EOS")[:-1]), st_row, tokenizer, strategy=strategy, add_situ=add_situ)
        # process output (decoder input) text
        d_inputs, d_roles, d_turns, d_strategy_labels, emotion = self._get_inputs_from_text(row.split("EOS")[-1], st_row, tokenizer, strategy=True, add_situ=False, sos=False)
        
        
        # print("EOS".join(row.split("EOS")[:-1]))
        # print(row.split("EOS")[-1])
        # raise Exception("debug")

        # make feature for input text
        eos = tokenizer.encode("EOS", add_special_tokens=False)[0]
        feature = self._make_feature(idx, inputs, roles, turns, eos, pad=pad, strategy_labels=strategy_labels, evaluate=evaluate, str_embd=True, generation=generation)
        # make feature for output (decoder input) text
        d_feature = self._make_feature(idx, d_inputs, d_roles, d_turns, eos, pad=pad, strategy_labels=d_strategy_labels, evaluate=evaluate, str_embd=True, generation=generation)
        comet_ids, comet_mask = self._get_comet_input(comet_row, tokenizer)
        comet_st_ids, comet_st_mask = self._get_comet_input(comet_st_row, tokenizer, max_num_attr=20)
        feature = InputFeatures_blender(feature, d_feature, comet_ids, comet_mask, emotion, comet_st_ids, comet_st_mask)
        return feature



    def _get_comet_input(self, comet_row, tokenizer, max_num_attr=30, max_len_attr=10):
        attrs = comet_row.split('__EOS__')[:-1]
        comet_ids = []
        comet_mask = []
        for ids, attr in enumerate(attrs):
            if ids == max_num_attr:
                break
            comet_attr_ids = tokenizer.encode(attr)
            if len(comet_attr_ids) < max_len_attr:
                comet_attr_ids += [tokenizer.pad_token_id]*(max_len_attr - len(comet_attr_ids)) 
            else:
                comet_attr_ids = comet_attr_ids[:max_len_attr]
            comet_ids.append(comet_attr_ids)
            comet_mask.append(1)

        if len(comet_ids) < max_num_attr:
            comet_ids += ([[tokenizer.pad_token_id]*max_len_attr]) * (max_num_attr - len(comet_ids))
            comet_mask += [0] * (max_num_attr - len(comet_mask))
        # print(attrs) 
        # print(comet_ids)
        # print(comet_mask)
        # print(error)
        
        assert len(comet_ids) == max_num_attr
        assert len(comet_mask) == max_num_attr
        return comet_ids, comet_mask

    def _get_inputs_from_text(self, text, st_row, tokenizer, strategy=False, cls = False, add_gen=False, add_situ=False, sos=True):
        srcs = text.strip()
        inputs = []
        roles = []
        turns = []
        strategy_labels=[]

        # adding tokens before encoding
        # srcs += " [GEN]"
        # srcs += " Response:"
        # if add_gen: srcs += " [GEN]"
        if add_gen: srcs += " Response:"
        if add_situ: srcs = "0 0 0 Context:" + st_row.strip() + " EOS" + srcs

        srcs = srcs.split(" EOS")
        emotion = None
        for idx, src in enumerate(srcs):

            if src =="":
                continue
            src_emo, src_role, src_turn, src = self._norm_text(src)
            if emotion is None:
                emotion = src_emo

            context_id = tokenizer.encode(src + " EOS", add_special_tokens=False)
            if not sos: context_id = context_id[1:]

            if not strategy:
                context_id = [i for i in context_id if i not in self.strategyIds]
            elif cls:
                context_id = tokenizer.cls + [i for i in context_id if i not in self.strategyIds]
            else:
                pass

            if src_role==1:
                try:
                    label = "["+src.split("[")[1].split("]")[0]+"]"
                except Exception as e:
                    strategy_labels.append(self.strategyNull)
                else:
                    strategy_labels.append(tokenizer.encode(label, add_special_tokens=False)[0])
                    # print(tokenizer.decode(self.strategyNull))
                    # raise Exception("debug")
            else:
                strategy_labels.append(self.strategyNull)

            inputs.append(context_id)
            roles.append(src_role)
            turns.append(src_turn)

        return inputs, roles, turns, strategy_labels, emotion


    def _make_feature(self, id_, sents, rls, ts, eos, pad=False, block_size=512, strategy_labels=None, evaluate=False, str_embd=False, generation=False):
        # we did't use role label and turn number in modeling as they did't carry significant improvement. However, codes still remain here.
        if len(sents) == 0:
            return InputFeatures_train([], [], [], [], [],
                                [], [] , [], [])
        input_ids = [i for s in sents for i in s]

        input_ids = input_ids
        lm_labels = []
        token_type_ids = []
        roles = []
        strategy_ids = []

        for i, s in enumerate(sents):
            token_type_ids += [ts[i]] * (len(s))
            flag_str = -1
            if str_embd: #use for strategy embed but currently we treat strategy as token
                strategy_ids += [strategy_labels[-1]] * (len(s))
            else:
                strategy_ids += [self.strategyNull] * (len(s))
            if i < len(sents) - 1:
                lm_labels += [-100] * (len(s))
                roles += [rls[i]] * (len(s))
            else:
                lm_labels += (  s )
                roles += [rls[i]] * (len(s))

        i = len(lm_labels) - 1
        if len(input_ids) == 1:
            print(input_ids, lm_labels, token_type_ids, roles)
        while i >= 0:
            if lm_labels[i] != -100:
                break
            i -= 1
        input_ids = input_ids[:i+1]
        lm_labels = lm_labels[:i+1]
        token_type_ids = token_type_ids[:i+1]
        roles = roles[:i+1]
        if not str_embd:
            strategy_ids = [self.strategyNull]*len(input_ids) # strategy is not used
        else:
            strategy_ids = strategy_ids[:i+1]
        if len(input_ids) == 1:
            print(input_ids, lm_labels, token_type_ids, roles)


        assert (len(input_ids) == len(token_type_ids)
                == len(lm_labels) == len(roles) == len(strategy_ids))
        # cut according to block size
        if len(input_ids) > block_size:
            cut_index = input_ids.index(eos,-512) + 1
            input_ids = input_ids[cut_index: ]

            token_type_ids = token_type_ids[cut_index: ]
            lm_labels = lm_labels[cut_index: ]
            roles = roles[cut_index: ]
            strategy_ids = strategy_ids[cut_index: ]
        # pad to multiples of 8
        if pad:
            while len(input_ids) % 8 != 0:
                input_ids.append(self.tokenizer.pad_token_id)
                token_type_ids.append(self.tokenizer.pad_token_id)
                lm_labels.append(-100)
                roles.append(self.tokenizer.pad_token_id)
                strategy_ids.append(self.strategyNull)
            assert len(input_ids) % 8 == 0
        position_ids = list(range(len(input_ids)))
        assert (len(input_ids) == len(position_ids) == len(token_type_ids)
                == len(lm_labels) == len(roles) == len(strategy_ids))
        if len(input_ids) == 0:
            import pdb
            pdb.set_trace()
        elif len(input_ids) == 1:
            print(input_ids, lm_labels, token_type_ids, roles)
        if True:
            # if it is for generation, the last sentence of context is the last sentence
            cls_position = len(input_ids)-1-input_ids[::-1].index(eos)
        else:
            # if not, the last sentence of context is the second last sentence
            cls_position = len(input_ids)-1-input_ids[::-1].index(eos,input_ids[::-1].index(eos)+1)
        
        # if evaluate and strategy_labels[-1]!=self.strategyNull:
        #     try:
        #         lm_labels[lm_labels.index(strategy_labels[-1]+50257+4687)] = -100
        #     except Exception:
        #         pass

        feature = InputFeatures_train(id_, input_ids, position_ids, token_type_ids, roles,
                                lm_labels, cls_position , strategy_labels[-1], strategy_ids)
        return feature

    def _norm_text(self, text):
        emo, r, t, *toks = text.strip().split()
        try:
            emo = int(emo)
            r = int(r)
            t = int(t)
            toks = ' '.join(toks[:len(toks)])
        except Exception as e:
            raise e
        return emo, r, t, toks



    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        return self.features[item]

    # @staticmethod
    def collate(self, features):
        input_ids = pad_sequence([torch.tensor(f.input_ids, dtype=torch.long)
                                  for f in features],
                                 batch_first=True, padding_value=self.tokenizer.pad_token_id)

        position_ids = pad_sequence([torch.tensor(f.position_ids,
                                                  dtype=torch.long)
                                     for f in features],
                                    batch_first=True, padding_value=self.tokenizer.pad_token_id)
        token_type_ids = pad_sequence([torch.tensor(f.token_type_ids,
                                                    dtype=torch.long)
                                       for f in features],
                                      batch_first=True, padding_value=self.tokenizer.pad_token_id)
        role_ids = pad_sequence([torch.tensor(f.role_ids, 
                                              dtype=torch.long)
                                      for f in features],
                                     batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence([torch.tensor(f.lm_labels, dtype=torch.long)
                               for f in features],
                              batch_first=True, padding_value=-100)
        
        cls_positions = torch.tensor([f.cls_position for f in features], dtype=torch.long)
        
        cls_labels = torch.tensor([f.cls_label for f in features], dtype=torch.long)
        
        strategy_ids = pad_sequence([torch.tensor(f.strategy_ids, dtype=torch.long)
                               for f in features],
                              batch_first=True, padding_value=self.tokenizer.pad_token_id)

        decoder_input_ids = pad_sequence([torch.tensor(f.decoder_input_ids, dtype=torch.long)
                                  for f in features],
                                 batch_first=True, padding_value=self.tokenizer.pad_token_id)
        decoder_position_ids = pad_sequence([torch.tensor(f.decoder_position_ids,
                                                  dtype=torch.long)
                                     for f in features],
                                    batch_first=True, padding_value=self.tokenizer.pad_token_id)
        decoder_token_type_ids = pad_sequence([torch.tensor(f.decoder_token_type_ids,
                                                    dtype=torch.long)
                                       for f in features],
                                      batch_first=True, padding_value=self.tokenizer.pad_token_id)
        decoder_role_ids = pad_sequence([torch.tensor(f.decoder_role_ids,
                                              dtype=torch.long)
                                      for f in features],
                                     batch_first=True, padding_value=self.tokenizer.pad_token_id)
        decoder_labels = pad_sequence([torch.tensor(f.decoder_lm_labels, dtype=torch.long)
                               for f in features],
                              batch_first=True, padding_value=-100)

        decoder_cls_positions = torch.tensor([f.decoder_cls_position for f in features], dtype=torch.long)

        decoder_cls_labels = torch.tensor([f.decoder_cls_label for f in features], dtype=torch.long)

        decoder_strategy_ids = pad_sequence([torch.tensor(f.decoder_strategy_ids, dtype=torch.long)
                               for f in features],
                              batch_first=True, padding_value=self.tokenizer.pad_token_id)
        # print([f.comet_ids for f in features])
        # print([f.comet_mask for f in features])
        comet_ids = torch.tensor([f.comet_ids for f in features], dtype=torch.long)
        comet_mask = torch.tensor([f.comet_mask for f in features], dtype=torch.long)
        emotion = torch.tensor([f.emotion for f in features], dtype=torch.long)
        comet_st_ids = torch.tensor([f.comet_st_ids for f in features], dtype=torch.long)
        comet_st_mask = torch.tensor([f.comet_st_mask for f in features], dtype=torch.long)

        return (input_ids, position_ids, token_type_ids, role_ids, labels, cls_positions, cls_labels, strategy_ids, decoder_input_ids, decoder_position_ids, decoder_token_type_ids, decoder_role_ids, decoder_labels, decoder_cls_positions, decoder_cls_labels, decoder_strategy_ids, comet_ids, comet_mask, emotion, comet_st_ids, comet_st_mask)




class InputFeatures_train(object):
    def __init__(self, conv_id, input_ids, position_ids, token_type_ids,
                 role_ids, lm_labels, cls_position, cls_label, strategy_ids, input_len=None):
        self.conv_id = conv_id
        self.input_ids = input_ids
        self.position_ids = position_ids
        self.token_type_ids = token_type_ids
        self.role_ids = role_ids
        self.lm_labels = lm_labels
        self.cls_position = cls_position
        self.cls_label = cls_label
        self.strategy_ids = strategy_ids
        if input_len is None:
            self.input_len = len(input_ids)
        else:
            self.input_len = input_len


class InputFeatures_blender(object):
    def __init__(self, encoder_feature, decoder_feature, comet_ids, comet_mask, emotion, comet_st_ids, comet_st_mask, comet_by_step_ids=None, comet_by_step_mask=None):
        self.conv_id = encoder_feature.conv_id
        self.input_ids = encoder_feature.input_ids
        self.position_ids = encoder_feature.position_ids
        self.token_type_ids = encoder_feature.token_type_ids
        self.role_ids = encoder_feature.role_ids
        self.lm_labels = encoder_feature.lm_labels
        self.cls_position = encoder_feature.cls_position
        self.cls_label = encoder_feature.cls_label
        self.strategy_ids = encoder_feature.strategy_ids
        self.decoder_input_ids = decoder_feature.input_ids
        self.decoder_position_ids = decoder_feature.position_ids
        self.decoder_token_type_ids = decoder_feature.token_type_ids
        self.decoder_role_ids = decoder_feature.role_ids
        self.decoder_lm_labels = decoder_feature.lm_labels
        self.decoder_cls_position = decoder_feature.cls_position
        self.decoder_cls_label = decoder_feature.cls_label
        self.decoder_strategy_ids = decoder_feature.strategy_ids
        self.comet_ids = comet_ids
        self.comet_mask = comet_mask
        self.emotion = emotion
        self.comet_st_ids = comet_st_ids
        self.comet_st_mask = comet_st_mask
        self.comet_by_step_ids = comet_by_step_ids
        self.comet_by_step_mask = comet_by_step_mask


def read_data_files(args, split='eval'):
    with open(args.data_path+"/"+args.__getitem__(f"{split}_file_name"),"r", encoding="utf-8") as f:
        chat_texts = f.read().split("\n")
    with open(args.data_path+"/" + args.__getitem__(f"situation_{split}_file_name"), "r", encoding="utf-8") as f:
        st_texts = f.read().split("\n")
    with open(args.data_path+"/"+ args.__getitem__(f"{split}_comet_file"), "r", encoding="utf-8") as f:
        comet = f.read().split("\n")
    with open(args.data_path+"/"+ args.__getitem__(f"situation_{split}_comet_file"), "r", encoding="utf-8") as f:
        comet_st = f.read().split("\n")
    with open(args.data_path+"/"+ args.__getitem__(f"cometStep_{split}_file_name"), "r", encoding="utf-8") as f:
        # this is a jsonl file
        comet_by_step = f.read().split("\n")
    return chat_texts, st_texts, comet, comet_st, comet_by_step
