from collections import defaultdict
from typing import Dict, List, Tuple, Union, overload

import torch
from fitbert.delemmatize import Delemmatizer
from fitbert.utils import mask as _mask
from functional import pseq, seq
from transformers import (
    BertForMaskedLM,
    BertTokenizer,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    XLNetLMHeadModel,
    XLNetTokenizer,
)


class FitBertNew:
    def __init__(
        self,
        model=None,
        model_xlnet=None,
        tokenizer=None,
        tokenizer_xlnet=None,
        model_name="bert-large-uncased",
        model_name_xlnet="xlnet-base-cased",
        mask_token="***mask***",
        disable_gpu=False,
        ensemble=False,
    ):
        self.mask_token = mask_token
        self.delemmatizer = Delemmatizer()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not disable_gpu else "cpu"
        )
        print("device:", self.device)
        self.ensemble = ensemble

        if not model:
            print("using model:", model_name)
            if "distilbert" in model_name:
                self.bert = DistilBertForMaskedLM.from_pretrained(model_name)
            else:
                self.bert = BertForMaskedLM.from_pretrained(model_name)
            self.bert.to(self.device)
        else:
            print("using custom model:", model.config.architectures)
            self.bert = model
            self.bert.to(self.device)
            
        if not tokenizer:
            if "distilbert" in model_name:
                self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
            else:
                self.tokenizer = BertTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = tokenizer
        
        if ensemble:
            if not model_xlnet:
                self.xlnet = XLNetLMHeadModel.from_pretrained(model_name_xlnet, mem_len=1024)
                self.xlnet.to(self.device)
                
            if not tokenizer:
                self.xlnet_tokenizer = XLNetTokenizer.from_pretrained(model_name_xlnet)
            self.xlnet.eval()

        self.bert.eval()

    @staticmethod
    def softmax(x):
        return x.exp() / (x.exp().sum(-1)).unsqueeze(-1)

    @staticmethod
    def is_multi(options: List[str]) -> bool:
        return seq(options).filter(lambda x: len(x.split()) != 1).non_empty()

    def mask(self, s: str, span: Tuple[int, int]) -> Tuple[str, str]:
        return _mask(s, span, mask_token=self.mask_token)

    def _tokens_to_masked_ids(self, tokens, mask_ind):
        masked_tokens = tokens[:]
        masked_tokens[mask_ind] = "[MASK]"
        masked_tokens = ["[CLS]"] + masked_tokens + ["[SEP]"]
        masked_ids = self.tokenizer.convert_tokens_to_ids(masked_tokens)
        return masked_ids

    def _get_sentence_probability(self, sent: str) -> float:

        tokens = self.tokenizer.tokenize(sent)
        input_ids = (
            seq(tokens)
            .enumerate()
            .starmap(lambda i, x: self._tokens_to_masked_ids(tokens, i))
            .list()
        )

        tens = torch.tensor(input_ids).to(self.device)
        with torch.no_grad():
            preds = self.bert(tens)[0]
            probs = self.softmax(preds)
            tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            prob = (
                seq(tokens_ids)
                .enumerate()
                .starmap(lambda i, x: float(probs[i][i + 1][x].item()))
                .reduce(lambda x, y: x * y, 1)
            )

            del tens, preds, probs, tokens, input_ids
            if self.device == "cuda":
                torch.cuda.empty_cache()

            return prob

    def _delemmatize_options(self, options: List[str]) -> List[str]:
        options = (
            seq(options[:])
            .flat_map(lambda x: self.delemmatizer(x))
            .union(options)
            .list()
        )
        return options

    def rank_single(self, masked_sent: str, words: List[str]):

        pre, post = masked_sent.split(self.mask_token)

        tokens = ["[CLS]"] + self.tokenizer.tokenize(pre)
        target_idx = len(tokens)
        tokens += ["[MASK]"]
        tokens += self.tokenizer.tokenize(post) + ["[SEP]"]

        words_ids = (
            seq(words)
            .map(lambda x: self.tokenizer.tokenize(x))
            .map(lambda x: self.tokenizer.convert_tokens_to_ids(x)[0])
        )

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        tens = torch.tensor(input_ids).unsqueeze(0)
        tens = tens.to(self.device)
        if not self.ensemble:
            with torch.no_grad():
                preds = self.bert(tens)[0]
                probs = self.softmax(preds)

                ranked_pairs = (
                    seq(words_ids)
                    .map(lambda x: float(probs[0][target_idx][x].item()))
                    .zip(words)
                    .sorted(key=lambda x: x[0], reverse=True)
                )

                ranked_options = (seq(ranked_pairs).map(lambda x: x[1])).list()
                ranked_options_prob = (seq(ranked_pairs).map(lambda x: x[0])).list()

                del tens, preds, probs, tokens, words_ids, input_ids
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                return ranked_options, ranked_options_prob
        else:
            tokens_xlnet = ["[CLS]"] + self.xlnet_tokenizer.tokenize(pre)
            target_idx_xlnet = len(tokens_xlnet)
            tokens_xlnet += ["[MASK]"]
            tokens_xlnet += self.xlnet_tokenizer.tokenize(post) + ["[SEP]"]

            words_ids_xlnet = (
              seq(words)
              .map(lambda x: self.xlnet_tokenizer.tokenize(x))
              .map(lambda x: self.xlnet_tokenizer.convert_tokens_to_ids(x)[0])
            )

            input_ids_xlnet = self.xlnet_tokenizer.convert_tokens_to_ids(tokens_xlnet)
            tens_xlnet = torch.tensor(input_ids_xlnet).unsqueeze(0)
            tens_xlnet = tens_xlnet.to(self.device)

            with torch.no_grad():
              # BERT STARTS HERE
              
              preds = self.bert(tens)[0]
              probs = self.softmax(preds)

              probs_bert = seq(words_ids).map(lambda x: float(probs[0][target_idx][x].item()))

              probs_bert = probs_bert.to_list()
              probs_bert = [float(i)/sum(probs_bert) for i in probs_bert]

              # BERT ENDS HERE

              # XLNET STARTS HERE

              preds = self.xlnet(tens_xlnet)[0]
              probs = self.softmax(preds)

              probs_xlnet = seq(words_ids_xlnet).map(lambda x: float(probs[0][target_idx_xlnet][x].item()))

              probs_xlnet = probs_xlnet.to_list()
              probs_xlnet = [float(i)/sum(probs_xlnet) for i in probs_xlnet]

              # XLNET ENDS HERE

              norm_probs = seq([(probs_bert[i] + probs_xlnet[i])/2 for i in range(len(probs_bert))])

              ranked_pairs = (
                  norm_probs.zip(words)
                  .sorted(key=lambda x: x[0], reverse=True)
              )

              ranked_options = (seq(ranked_pairs).map(lambda x: x[1])).list()
              ranked_options_prob = (seq(ranked_pairs).map(lambda x: x[0])).list()

              del tens, preds, probs, tokens, words_ids, input_ids, tens_xlnet, tokens_xlnet, words_ids_xlnet, input_ids_xlnet, norm_probs, probs_xlnet, probs_bert
              if self.device == "cuda":
                  torch.cuda.empty_cache()
              return ranked_options, ranked_options_prob

    def _simplify_options(self, sent: str, options: List[str]):

        options_split = seq(options).map(lambda x: x.split())

        trans_start = list(zip(*options_split))

        start = (
            seq(trans_start)
            .take_while(lambda x: seq(x).distinct().len() == 1)
            .map(lambda x: x[0])
            .list()
        )

        options_split_reversed = seq(options_split).map(
            lambda x: seq(x[len(start) :]).reverse()
        )

        trans_end = list(zip(*options_split_reversed))

        end = (
            seq(trans_end)
            .take_while(lambda x: seq(x).distinct().len() == 1)
            .map(lambda x: x[0])
            .list()
        )

        start_words = seq(start).make_string(" ")
        end_words = seq(end).reverse().make_string(" ")

        options = (
            seq(options_split)
            .map(lambda x: x[len(start) : len(x) - len(end)])
            .map(lambda x: seq(x).make_string(" ").strip())
            .list()
        )

        sub = seq([start_words, self.mask_token, end_words]).make_string(" ").strip()
        sent = sent.replace(self.mask_token, sub)

        return options, sent, start_words, end_words

    def rank(
        self,
        sent: str,
        options: List[str],
        delemmatize: bool = False,
        with_prob: bool = False,
    ):
        """
        Rank a list of candidates

        returns: Either a List of strings,
        or if `with_prob` is True, a Tuple of List[str], List[float]

        """

        options = seq(options).distinct().list()

        if delemmatize:
            options = seq(self._delemmatize_options(options)).distinct().list()

        if seq(options).len() == 1:
            return options
        
        options, sent, start_words, end_words = self._simplify_options(sent, options)

        if self.is_multi(options):
            ranked, prob = self.rank_multi(sent, options)
        else:
            ranked, prob = self.rank_single(sent, options)

        ranked = (
            seq(ranked)
            .map(lambda x: [start_words, x, end_words])
            .map(lambda x: seq(x).make_string(" ").strip())
            .list()
        )
        if with_prob:
            return ranked, prob
        else:
            return ranked
