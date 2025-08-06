# -*- coding: utf-8 -*-
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import sys
import time
import progressbar
import gensim
from util.file_util import load_data, get_stem
from util.fast_clustering import community_detection

sys.path.append("..")
from app_cfg import use_cuda, llm_atep

if not use_cuda:
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")


def upd_clusters(ori_clusters):
    new_clusters = list()
    other_comments = list()
    for cluster in ori_clusters:
        if len(cluster) == 1:
            other_comments += cluster
        else:
            new_clusters.append(cluster)
    return new_clusters, other_comments


class Trainer:
    def __init__(self, data_set, comment_num, clustering_thr_, background_info):
        self.tokenizer = AutoTokenizer.from_pretrained('ml_models/sent_bert', padding_side='left')
        self.model = AutoModel.from_pretrained('ml_models/sent_bert')
        self.model.to(device)
        self.train_set = data_set
        self.clustering_thr = clustering_thr_
        self.comment_num = comment_num
        self.bg_info = None if background_info is None else get_stem(background_info)

    def comment_encoding(self, comments):
        tr_cmts = list()
        tr_cmts_count = list()
        for cmt in comments:
            if len(cmt.split()) < 400:
                tr_cmts_count.append(1)
                tr_cmts.append(cmt)
            else:
                tr_cmt_one = ""
                tr_cmt_one_len = 0
                tr_cmt_one_count = 0
                doc = nlp(cmt)
                for sent in doc.sents:
                    sent_one = str(sent)
                    sent_len = len(sent_one.split())
                    if sent_len + tr_cmt_one_len >= 400:
                        tr_cmt_one_count += 1
                        tr_cmts.append(tr_cmt_one.strip())
                        tr_cmt_one = sent_one
                        tr_cmt_one_len = sent_len
                    else:
                        tr_cmt_one = tr_cmt_one + " " + sent_one
                        tr_cmt_one_len += sent_len
                if len(tr_cmt_one.strip()) > 0:
                    tr_cmt_one_count += 1
                    tr_cmts.append(tr_cmt_one.strip())
                tr_cmts_count.append(tr_cmt_one_count)
        # encoding
        corpus_embeddings = self.ml_encoding(tr_cmts)
        sent_idx = 0
        corpus_embeddings_ = None
        for sent_count in tr_cmts_count:
            embedding_one = corpus_embeddings[sent_idx] if sent_count == 1 else torch.mean(corpus_embeddings[sent_idx:sent_count, :], dim=0)
            embedding_one = embedding_one.unsqueeze(0)
            corpus_embeddings_ = embedding_one if corpus_embeddings_ is None else torch.cat((corpus_embeddings_, embedding_one), 0)
            sent_idx += sent_count
        return corpus_embeddings_

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def ml_encoding(self, sentences):
        """ Encoding all sentences will cause OOM issue, how about batch encoding? Reduce the Padding risk.
            Encode every 'speed' sentences
        """
        sent_num = len(sentences)
        begin_, speed, end_ = 0, 200, min(200, sent_num)
        embeddings_all = None
        while True:
            sent_batch = sentences[begin_: end_]
            encoded_input = self.tokenizer(sent_batch, padding=True, truncation=True, return_tensors='pt').to(device)
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            embeddings_all = sentence_embeddings if embeddings_all is None else torch.cat((embeddings_all, sentence_embeddings), 0)
            if end_ >= sent_num:
                break
            else:
                begin_ = end_
                end_ = min(sent_num, end_ + speed)
        return embeddings_all

    def train_plain(self, logger):
        # cmt_vec = None
        keywords_all = list()
        corpus_sentences = list()
        sentences_cmt_ids = list()

        for item in self.train_set:
            keywords, text_item, text_cmt_id = item
            keywords_all.append(keywords)
            # when the title is a none string, try using the aspect terms instead, in other tasks, the input cannot be "", no worry
            if text_item.strip() == "" and keywords is not None:
                corpus_sentences.append(", ".join(keywords))
            else:
                corpus_sentences.append(text_item)
            sentences_cmt_ids.append(text_cmt_id)

        logger.info("Encoding & clustering comments in an article cluster...")

        # ML sentence rep
        corpus_embeddings = self.ml_encoding(corpus_sentences)

        if llm_atep:
            keywords_all_ = ["Aspect Terms: " + ", ".join(keywords_) for keywords_ in keywords_all]
            kw_embeddings = self.ml_encoding(keywords_all_)
            corpus_embeddings = torch.cat((kw_embeddings, corpus_embeddings), -1)

        corpus_word_stems = None if self.bg_info is None else [get_stem(sentence) for sentence in corpus_sentences]

        comment_num_c = corpus_embeddings.size()[0]

        if comment_num_c == 1:
            clusters = [[0]]
        else:
            thr_min = self.clustering_thr[0]
            thr_delta = self.clustering_thr[1]
            thr_est = self.clustering_thr[2]
            thr_est_num = self.clustering_thr[3]
            clusters = community_detection(corpus_embeddings, threshold=thr_min, max_delta_thr=thr_delta, estimate_thr=thr_est, estimate_num=thr_est_num, bg_stems=self.bg_info, sentences_stems=corpus_word_stems)

        clusters, other_comment_ids = upd_clusters(clusters)

        cluster_id = 0
        clusters2save = list()
        comment_labels = [-1 for _ in range(self.comment_num)]
        for cluster in clusters:
            comment_ids = cluster[:]
            selected_ids = list()
            kw_list, sentences, sentences_ids = list(), list(), list()
            for sent_id in comment_ids:
                if llm_atep and len(keywords_all[sent_id]) == 0:
                    other_comment_ids.append(sent_id)
                else:
                    comment_labels[sent_id] = cluster_id
                    kw_list.append(keywords_all[sent_id])
                    sentences.append(corpus_sentences[sent_id])
                    sentences_ids.append(sentences_cmt_ids[sent_id])
                    selected_ids.append(sent_id)

            if len(selected_ids) > 1:
                clusters2save.append((sentences, kw_list, sentences_ids))
                cluster_id += 1
            else:
                other_comment_ids += selected_ids

        # build the other comment cluster
        kw_list, sentences, sentences_ids = list(), list(), list()

        for idx_flag, sent_id in enumerate(other_comment_ids):
            comment_labels[sent_id] = cluster_id
            kw_list.append(keywords_all[sent_id])
            if idx_flag == 0:
                sentences.append("$$Other comments.$$ " + corpus_sentences[sent_id])
            else:
                sentences.append(corpus_sentences[sent_id])
            sentences_ids.append(sentences_cmt_ids[sent_id])

        clusters2save.append((sentences, kw_list, sentences_ids))
        assert -1 not in comment_labels
        return clusters2save, comment_labels

    def train_plain_wokw(self, logger):
        """ wokw: w/o keywords
        """
        sentences_cmt_ids = list()
        corpus_embeddings = None
        # here text_item can contain many aspects
        n_tracks = len(self.train_set[0][0])
        corpus_sentences = [[] for _ in range(n_tracks)]
        for item in self.train_set:
            text_item, text_cmt_id = item
            for idx_, event_aspect in enumerate(text_item):
                corpus_sentences[idx_] = corpus_sentences[idx_] + [event_aspect]
            sentences_cmt_ids.append(text_cmt_id)
        logger.info("Encoding & clustering comments in an article cluster...")

        # ML sentence rep
        corpus_embeddings = None
        for idx_ in range(n_tracks):
            corpus_embeddings_new = self.ml_encoding(corpus_sentences[idx_])
            if corpus_embeddings is None:
                corpus_embeddings = self.ml_encoding(corpus_sentences[idx_])
            else:
                corpus_embeddings = torch.cat((corpus_embeddings, corpus_embeddings_new), dim=-1)
        corpus_word_stems = None  # After considering ME, I remove this for clear usage
        comment_num_c = corpus_embeddings.size()[0]
        if comment_num_c == 1:
            clusters = [[0]]
        else:
            thr_min = self.clustering_thr[0]
            thr_delta = self.clustering_thr[1]
            thr_est = self.clustering_thr[2]
            thr_est_num = self.clustering_thr[3]
            clusters = community_detection(corpus_embeddings, threshold=thr_min, max_delta_thr=thr_delta, estimate_thr=thr_est, estimate_num=thr_est_num, bg_stems=self.bg_info, sentences_stems=corpus_word_stems)
        clusters, other_comment_ids = upd_clusters(clusters)
        cluster_id = 0
        clusters2save = list()
        comment_labels = [-1 for _ in range(self.comment_num)]
        for cluster in clusters:
            comment_ids = cluster[:]
            selected_ids = list()
            sentences, sentences_ids = list(), list()
            for sent_id in comment_ids:
                comment_labels[sent_id] = cluster_id
                # sentences.append(corpus_sentences[sent_id])
                sentences_ids.append(sentences_cmt_ids[sent_id])
                selected_ids.append(sent_id)
            if len(selected_ids) > 1:
                clusters2save.append((sentences, sentences_ids))
                cluster_id += 1
            else:
                other_comment_ids += selected_ids

        # build the other comment cluster
        sentences, sentences_ids = list(), list()

        if len(other_comment_ids) == 0:
            clusters2save.append(([], []))
        else:
            for idx_flag, sent_id in enumerate(other_comment_ids):
                comment_labels[sent_id] = cluster_id
                # if idx_flag == 0:
                #     sentences.append("$$Other comments.$$ " + corpus_sentences[sent_id])
                # else:
                #     sentences.append(corpus_sentences[sent_id])
                sentences_ids.append(sentences_cmt_ids[sent_id])

            clusters2save.append((sentences, sentences_ids))
        assert -1 not in comment_labels
        return clusters2save, comment_labels

    def train(self, logger, wokw=False):
        if wokw:
            clusters2save, comment_labels = self.train_plain_wokw(logger)
        else:
            clusters2save, comment_labels = self.train_plain(logger)
        return clusters2save, comment_labels
