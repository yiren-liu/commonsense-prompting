import json

import warnings
import numpy as np
import nltk
from typing import List
from collections import Counter
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from datasets import load_dataset, load_metric

class Metric(object):
    def __init__(self, toker, hyp_path, ref_path):
        self.refs = []
        self.hyps = []
        with open(hyp_path, 'r', encoding='utf-8') as f:
            hyps = json.load(f)
        with open(ref_path, 'r', encoding='utf-8') as f:
            refs = json.load(f)
        assert len(hyps) == len(refs)
        self.toker = toker
        for i in range(len(hyps)):
            self.forword([refs[i]], hyps[i])

        

    def forword(self, refs: str, hyp: str, chinese=False): # TODO: only applicable to English
        # if not chinese:
        #     self.refs.append([nltk.word_tokenize(e.lower()) for e in refs])
        #     self.hyps.append(nltk.word_tokenize(hyp.lower()))
        # else:
        self.refs.append([self.toker.tokenize(e) for e in refs])
        self.hyps.append(self.toker.tokenize(hyp))
        # print(self.refs)
        # print(self.hyps)
        # return     
    def calc_bleu_k(self, k):
        weights = [1. / k] * k + (4 - k) * [0.]
        try:
            bleu = corpus_bleu(self.refs, self.hyps, weights=weights,
                               smoothing_function=SmoothingFunction().method3)
        except ZeroDivisionError as _:
            warnings.warn('the bleu is invalid')
            bleu = 0.
        return bleu

    def calc_distinct_k(self, k):
        d = {}
        tot = 0
        for sen in self.hyps:
            for i in range(0, len(sen)-k):
                key = tuple(sen[i:i+k])
                d[key] = 1
                tot += 1
        if tot > 0:
            dist = len(d) / tot
        else:
            warnings.warn('the distinct is invalid')
            dist = 0.
        return dist
    
    def calc_unigram_f1(self):
        f1_scores = []
        for hyp, refs in zip(self.hyps, self.refs):
            scores = []
            for ref in refs:
                cross = Counter(hyp) & Counter(ref)
                cross = sum(cross.values())
                p = cross / max(len(hyp), 1e-10)
                r = cross / max(len(ref), 1e-10)
                f1 = 2 * p * r / max(p + r, 1e-10)
                scores.append(f1)
            f1_scores.append(max(scores))
        return np.mean(f1_scores), f1_scores
    
    def calc_rouge_l(self, beta=1.2):
        scores = []
        for hyp, refs in zip(self.hyps, self.refs):
            prec = []
            rec = []
            for ref in refs:
                lcs = my_lcs(ref, hyp)
                prec.append(lcs / max(len(hyp), 1e-10))
                rec.append(lcs / max(len(ref), 1e-10))
            prec_max = max(prec)
            rec_max = max(rec)
            if prec_max != 0 and rec_max !=0:
                score = ((1 + beta**2)*prec_max*rec_max)/float(rec_max + beta**2*prec_max)
            else:
                score = 0.0
            scores.append(score)
        return np.mean(scores), scores
    
    def calc_meteor(self):
        meteor = load_metric("meteor")
        return meteor.compute(predictions=self.hyps, references=self.refs)

    def calc_comet(self):
        comet = load_metric("comet")
        return comet.compute(predictions=self.hyps, references=self.refs)
    
    def calc_bertscore(self):
        bertscore = load_metric("bertscore")
        return bertscore.compute(predictions=self.hyps, references=self.refs, lang="en")

    def calc_strategy_acc(self):
        strategy_acc = load_metric("accuracy")
        strategy_f1 = load_metric("f1")

        strategy_hyps = [tokens[0] for tokens in self.hyps]
        strategy_refs = [tokens[0] for ref in self.refs for tokens in ref]
        # build a index
        strategies = set(strategy_hyps + strategy_refs)
        strategy2idx = {s: i for i, s in enumerate(strategies)}
        strategy_hyps = [strategy2idx[s] for s in strategy_hyps]
        strategy_refs = [strategy2idx[s] for s in strategy_refs]

        res = {}
        res["strategy_acc"] = strategy_acc.compute(
            predictions=strategy_hyps, references=strategy_refs
        )
        res["strategy_f1"] = strategy_f1.compute(
            predictions=strategy_hyps, references=strategy_refs, average="micro"
        )
        return res



    def close(self):
        result = {
            'length': float(np.mean(list(map(len, self.hyps)))),
            **{f"dist-{k}": 100 * self.calc_distinct_k(k) for k in range(1, 4)},
            **{f"bleu-{k}": 100 * self.calc_bleu_k(k) for k in range(1, 5)}
        }
        
        f1, scores = self.calc_unigram_f1()
        result['f1'] = 100 * f1
        result_list = {
            'f1': scores
        }
        
        rl, scores = self.calc_rouge_l()
        result['rouge-l'] = 100 * rl
        result_list.update({
            'rouge-l': scores
        })

        meteor = self.calc_meteor()
        result['meteor'] = 100 * meteor['meteor']
        result_list.update({
            'meteor': meteor['meteor']
        })


        # comet = self.calc_comet()
        # result['comet'] = 100 * comet['comet']
        # result_list.update({
        #     'comet': comet['comet']
        # })

        bertscore = self.calc_bertscore()
        result['bertscore'] = 100 * np.mean(bertscore['f1'])
        result_list.update({
            'bertscore': np.mean(bertscore['f1'])
        })

        strategy_acc = self.calc_strategy_acc()
        result['strategy_acc'] = 100 * strategy_acc['strategy_acc']['accuracy']
        result['strategy_f1'] = 100 * strategy_acc['strategy_f1']['f1']
        result_list.update({
            'strategy_acc': strategy_acc['strategy_acc'],
            'strategy_f1': strategy_acc['strategy_f1']
        })

        return result, result_list


def my_lcs(string, sub):
    """
    Calculates longest common subsequence for a pair of tokenized strings
    :param string : list of str : tokens from a string split using whitespace
    :param sub : list of str : shorter string, also split using whitespace
    :returns: length (list of int): length of the longest common subsequence between the two strings

    Note: my_lcs only gives length of the longest common subsequence, not the actual LCS
    """
    if len(string) < len(sub):
        sub, string = string, sub

    lengths = [[0 for _ in range(0,len(sub)+1)] for _ in range(0,len(string)+1)]

    for j in range(1,len(sub)+1):
        for i in range(1, len(string) + 1):
            if string[i - 1] == sub[j - 1]:
                lengths[i][j] = lengths[i-1][j-1] + 1
            else:
                lengths[i][j] = max(lengths[i-1][j] , lengths[i][j-1])

    return lengths[len(string)][len(sub)]

def summary(test_file_path, generate_file_path, reference_file_path, summary_file_path, chat_texts, test_situation_file_path):
    with open(test_file_path, "r", encoding="utf-8") as f:
        ctx = f.read().split("\n")
    with open(test_situation_file_path, "r", encoding="utf-8") as f:
        st = f.read().split("\n")
    ctx = ctx[:-1]
    st = st[:-1]
    with open(generate_file_path, "r", encoding="utf-8") as f:
        gen_rep = json.load(f)
    with open(reference_file_path, "r", encoding="utf-8") as f:
        ref_rep = json.load(f)
    with open(summary_file_path, 'w', encoding='utf-8') as f:
        for (ctx_row, ref_rep_row, gen_rep_row, chat_text, st_row) in zip(ctx, ref_rep, gen_rep, chat_texts, st):
            query = process_row_to_comet_query(chat_text)
            if query is None:
                query = ""
            line = '[contxt]\t' + ctx_row + '\n[reference_response]\t' + ref_rep_row + '\n[hypothesis_response]\t' + gen_rep_row + '\n[comet query]\t' + query +'\n[situation]\t' + st_row +  '\n' * 2
            f.writelines(line)

def process_row_to_comet_query(row):
    sents = row.strip().split('EOS')
    n_sent = len(sents)
    all_seeker_uttrs = []
    for i in range(n_sent-1, -1, -1):
        # print(sents[i].strip().split(' '))
        tokens = sents[i].strip().split(' ')
        if int(tokens[1]) == 0:
            if int(tokens[1]) == 0:
                return ' '.join(tokens[3:])
                # all_seeker_uttrs.append(' '.join(tokens[3:]))
    # return '\t'.join(all_seeker_uttrs)
