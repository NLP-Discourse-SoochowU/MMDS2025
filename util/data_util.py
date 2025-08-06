"""
Author: Lynx Zhang
Date: 2020.
Email: zzlynx@outlook.com
"""
from sentence_transformers import SentenceTransformer, util
from util.file_util import *
import numpy as np
import progressbar
import torch
import pandas as pd
import spacy
import os
import time
import torch.nn as nn
import json
# from googletrans import Translator
import re
from tqdm import tqdm
from structure.comment_tree import comment_tree
from urllib.request import Request, urlopen
import random
line_id = 0
# translator = Translator()
# pretr_model = SentenceTransformer('all-MiniLM-L6-v2')
cosine_sim = nn.CosineSimilarity(dim=1)


def crawl_reddit_html():
    reddit_path = "data/reddit"
    for file_name in os.listdir(reddit_path):
        if not file_name.startswith("Xreddit"):
            continue
        file_path = os.path.join(reddit_path, file_name)
        with open(file_path, "r") as f:
            json_obj = json.load(f)
        articles = json_obj["data"]
        comments = []
        target_path = "data/html/" + file_name.split(".")[0]
        downloaded_files = os.listdir(target_path)
        for article in articles:
            if "comments" not in article.keys():
                continue
            c_url = article["c_url"]
            article_id = article["p_id"]

            # load data and write
            file_name_pre = article_id + ".html"
            if file_name_pre in downloaded_files:
                continue

            html_path = os.path.join(target_path, file_name_pre)

            req = Request(c_url)
            req.add_header('User-agent', 'your bot 0.1')
            html = urlopen(req).read()

            # response = urlopen()
            # html = response.read()
            write_over(html.decode("utf-8"), html_path)
            # time.sleep(5)
            print(file_name_pre)
        save_data(comments, creddit_path)


def build_reddit_trees():
    global line_id
    name2trees = dict()
    for dir_name in os.listdir("data/html"):
        dir_path = os.path.join("data/html", dir_name)
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            # comment_num = name2commnum[file_name.split(".")[0]]
            print("BEGIN")
            tree_lines = build_reddit_tree_one(file_path)
            tree_lines = tree_lines[1:]
            # initializa
            line_id = 0
            tree_objs = []
            print(file_path)
            while line_id < len(tree_lines):
                tree_obj = recursive_build(tree_lines)
                tree_obj.read_tree()
                tree_objs.append(tree_obj)
                line_id += 1
            name2trees[file_name[:-5]] = tree_objs  # article name to the comment trees
    save_data(name2trees, "data/annotations/name2trees.pkl")


def recursive_build(tree_lines, line_tag=-1):
    """ Depth-first
    """
    global line_id
    stack_val, comment_text_out = tree_lines[line_id]
    pattern = re.compile(r'.*?<p>(.*)</p>.*')
    comment_text_ = pattern.findall(comment_text_out)
    if len(comment_text_) == 0:
        comment_text_ = comment_text_out
    else:
        comment_text_dirty = comment_text_[0]
        comment_text_ = comment_text_dirty.replace("<p>", " ").replace("</p>", " ")
    if stack_val > line_tag:
        root = comment_tree(comment_text=comment_text_)

        # build children
        children = []
        while line_id + 1 < len(tree_lines):
            line_id += 1
            child = recursive_build(tree_lines, line_tag=stack_val)
            if child is not None:
                children.append(child)
            else:
                line_id -= 1
                break
        root.children = children
    else:
        # Do not have children any more
        root = None
    return root


def build_reddit_tree_one(file_path, comment_num=10000):
    """ Build comment tree objects.
        Only retain <div xxx> and </div> and <div class="md"> xxx </div>
    """
    tree_lines = []
    with open(file_path, "r") as f:
        lines = f.readlines()
    lines_new = [line.strip() for line in lines]
    text = " ".join(lines_new)
    pattern = re.compile(r'<div class="md">.*?</div>|<div.*?>|</div>')
    results = pattern.findall(text)
    pattern_comment = re.compile(r'<div class=" thing id-.*?; comment ".*?>')
    begin_flag = True
    stack = 0
    for result in results:
        # if pattern_comment.match(result) is not None:
        #     begin_flag = True
        if begin_flag:
            # count
            if result.startswith('<div class="md">'):
                if "This is an archived post." not in result:
                    tree_lines.append((stack, result))
                # print(stack, result)
            elif result.startswith('<div'):
                stack += 1
            else:
                stack -= 1
    return tree_lines


def build_tree_comment_db(tree_dict, carticle_name2trees):
    comment_tree_db = list()
    auto_comment_id_ = 0
    for article_id in tree_dict.keys():
        trees = tree_dict[article_id]
        for tree_id, tree in enumerate(trees):
            comments = tree.get_tree_comments(article_id, tree_id)
            for comment in comments:
                parts = comment.split("$^$")
                tree_comment_feat = "$^$".join([parts[0], parts[1], parts[2], str(auto_comment_id_)])
                comment_txt = parts[3]

                comment_dict = dict()
                comment_dict["cmt_id"] = auto_comment_id_
                comment_dict["cmt_article_id"] = article_id
                comment_dict["cmt_content"] = comment_txt
                comment_dict["cmt_tree"] = tree_comment_feat
                comment_tree_db.append(comment_dict)
                auto_comment_id_ += 1
    # dict 2 csv
    pd.DataFrame(comment_tree_db).to_csv(carticle_name2trees, index=None)


def build_ac_csv(data_source, article_cluster_db):
    ac_names = load_data(data_source)
    article_clusters = list()
    auto_ac_id = 0
    for ac_ in ac_names:
        ac_dict = dict()
        ac_dict["ac_id"] = auto_ac_id
        ac_dict["article_ids"] = " ".join(ac_)
        auto_ac_id += 1
        article_clusters.append(ac_dict)
    pd.DataFrame(article_clusters).to_csv(article_cluster_db, index=None)


def build_tree_comments():
    trees = load_data("data/annotations/name2trees.pkl")
    article_trees = list(trees.values())
    tree_names = list(trees.keys())
    name2comments = dict()
    for name, trees in zip(tree_names, article_trees):
        article_comments = []
        for tree_id, tree in enumerate(trees):
            comments = tree.get_tree_comments(name, tree_id)
            raw_comments = []
            for comment in comments:
                raw_comment = comment.split("$^$")[3]
                raw_comments.append(raw_comment)
            article_comments = article_comments + raw_comments
        name2comments[name] = article_comments
    save_data(name2comments, "data/annotations/name2comments.pkl")


def build_plain_comment_db(name2comments, carticle_name2comments):
    plain_comment_db = list()
    auto_comment_id_ = 0
    for article_id in name2comments.keys():
        art_comments = name2comments[article_id]
        for comment in art_comments:
            comment_dict = dict()
            comment_dict["cmt_id"] = auto_comment_id_
            comment_dict["cmt_article_id"] = article_id
            comment_dict["cmt_content"] = comment
            plain_comment_db.append(comment_dict)
            auto_comment_id_ += 1
    # dict 2 csv
    pd.DataFrame(plain_comment_db).to_csv(carticle_name2comments, index=None)


def build_tuning_data():
    """ Build the train and dev data for summary model fine-tuning.
        V1: Articles V2: Comment clusters
    """
    nlp = spacy.load('en_core_web_sm')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    article_sentences = list()
    article_cc_sentences = list()

    reddit_path = "data/reddit"
    for file_name in os.listdir(reddit_path):
        if file_name.startswith("reddit") and file_name != "reddit-05.json":
            file_path = os.path.join(reddit_path, file_name)
            with open(file_path, "r") as f:
                json_obj = json.load(f)
            articles = json_obj["data"]
            article_num = len(articles)
            for a_idx, article in enumerate(articles):
                if "comments" not in article.keys():
                    continue
                article_text = article["article"].strip()
                doc = nlp(article_text)
                sentences = " <S_SEP> ".join([str(sent).strip() for sent in doc.sents if len(str(sent).strip()) > 0])
                sentences = sentences.replace("\n", "")
                article_sentences.append(sentences)

                # special articles, for comment comtext understanding
                comments = article["comments"]
                comment_sentences = [[str(sent) for sent in doc.sents] for doc in [nlp(cmt.strip()) for cmt in comments]]
                if len(comment_sentences) < 2:
                    continue
                sentences = list()
                for sent_list in comment_sentences:
                    sentences += sent_list
                # rep & group them
                sent_embeddings = model.encode(sentences, batch_size=64, show_progress_bar=False, convert_to_tensor=True)
                groups = util.community_detection(sent_embeddings, min_community_size=4, threshold=0.55)
                print_("Clustering..." + str(a_idx + 1) + "/" + str(article_num) +
                       " sent_num: " + str(len(comment_sentences)) +
                       " group_num: " + str(len(groups)), "prep.txt")
                for i, group in enumerate(groups):
                    sent_ids = group[:]
                    cc_sentences = " <S_SEP> ".join([sentences[sent_id].strip() for sent_id in sent_ids if
                                                     len(sentences[sent_id].strip()) > 0])
                    if len(cc_sentences.strip()) > 0:
                        article_cc_sentences.append(cc_sentences)

        if file_name == "reddit-05.json":
            file_path = os.path.join(reddit_path, file_name)
            with open(file_path, "r") as f:
                json_obj = json.load(f)
            articles = json_obj["data"]
            dev_article_sentences = list()
            dev_article_cc_sentences = list()
            count = 0
            for a_idx, article in enumerate(articles):
                if count >= 1000:
                    break
                if "comments" not in article.keys():
                    continue
                article_text = article["article"].strip()
                doc = nlp(article_text)
                sentences = " <S_SEP> ".join([str(sent).strip() for sent in doc.sents if len(str(sent).strip()) > 0])
                sentences = sentences.replace("\n", "")
                dev_article_sentences.append(sentences)

                # special articles, for comment comtext understanding
                comments = article["comments"]
                comment_sentences = [[str(sent).strip() for sent in doc.sents] for doc in [nlp(cmt.strip()) for cmt in comments]]
                if len(comment_sentences) < 2:
                    continue
                sentences = list()
                for sent_list in comment_sentences:
                    sentences += sent_list
                # rep & group them
                sent_embeddings = model.encode(sentences, batch_size=64, show_progress_bar=False, convert_to_tensor=True)
                groups = util.community_detection(sent_embeddings, min_community_size=4, threshold=0.55)
                print_("Clustering..." + str(a_idx + 1) + "/" + str(500) +
                       " sent_num: " + str(len(comment_sentences)) +
                       " group_num: " + str(len(groups)), "prep.txt")
                if len(groups) == 0:
                    continue
                count += len(groups)
                for i, group in enumerate(groups):
                    sent_ids = group[:]
                    cc_sentences = " <S_SEP> ".join([sentences[sent_id].strip() for sent_id in sent_ids if
                                                     len(sentences[sent_id].strip()) > 0])
                    if len(cc_sentences.strip()) > 0:
                        dev_article_cc_sentences.append(cc_sentences)
            print(len(dev_article_sentences), len(dev_article_cc_sentences))
            write_iterate(dev_article_sentences, "data/tuning/valid.article")
            write_iterate(dev_article_cc_sentences, "data/tuning_cc/valid.article")

    print(len(article_sentences), len(article_cc_sentences))
    write_iterate(article_sentences, "data/tuning/train.article")
    write_iterate(article_cc_sentences, "data/tuning_cc/train.article")


def llama_instruct():
    """
    :return: Self-use for research only.
    """
    presentation = "data/uccs_test/presentation.txt"
    with open(presentation, "r") as f:
        lines = f.readlines()
    clusters = list()
    prompts = list()
    instructs = list()
    one_instruct = ""
    cc_flag = False
    cc_num = 1
    cc_count = 0
    for line in lines:
        if line.startswith("=============== comment cluster"):
            line = "=============== comment cluster " + "#" + str(cc_num) + " ==============="
            cc_num += 1
            cc_flag = True
            clusters.append(line)
            prompts.append(line)
            prompts.append("I have some sentences below:")
            one_instruct += "I have some sentences below: "
            cc_count = 0
            continue
        if cc_flag and not line.startswith("==============="):
            tokens = line.strip().split()
            new_line = tokens[0] + " " + " ".join(tokens[2:])
            clusters.append(new_line)
            prompts.append(new_line)
            one_instruct += (new_line + "\t")
            cc_count += 1
        if line.startswith("=============== summary"):
            if cc_flag:
                # clusters.pop(-1)
                clusters.append(line.strip())
                clusters.append("")
                # prompts.pop(-1)
                cc_count -= 1
                prompt_content = "Please abstract a summary for the sentences, the summary should not be longer than " +\
                                 str(min(max(1, cc_count-1), 3))\
                                 + " sentences, and the summary must be shorter than the original sentences."
                prompts.append(prompt_content)
                one_instruct += prompt_content
                instructs.append(one_instruct)
                one_instruct = ""
                prompts.append("")
                cc_flag = False

    # write_iterate(clusters, presentation)
    llama_target = "data/cache/presentation.txt"
    write_iterate(prompts, llama_target + ".prompt")
    write_iterate(instructs, llama_target + ".instructs")


def build_tuning_data_user(args):
    """ Build the train and dev data for summary model fine-tuning.
        Read from DB.
    """
    from db_conn import dsta_db
    db_obj = dsta_db(args.host, args.username, args.passwd, args.port, args.db_name)
    # db_obj.show_tables()
    article_data = db_obj.get_tuning_documents()
    train_set, dev_set = list(), list()
    for index, row in article_data.iterrows():
        if len(dev_set) >= 1000:
            train_set.append(row[1])
            continue
        if random.sample([True, False, False, False], 1)[0]:
            dev_set.append(row[1])
        else:
            train_set.append(row[1])

    nlp = spacy.load('en_core_web_sm')
    train_article_sentences = list()
    for article in train_set:
        article_text = article.strip()
        doc = nlp(article_text)
        sentences = " <S_SEP> ".join([str(sent).strip() for sent in doc.sents if len(str(sent).strip()) > 0])
        sentences = sentences.replace("\n", "")
        train_article_sentences.append(sentences)

    dev_article_sentences = list()
    for article in dev_set:
        article_text = article.strip()
        doc = nlp(article_text)
        sentences = " <S_SEP> ".join([str(sent).strip() for sent in doc.sents if len(str(sent).strip()) > 0])
        sentences = sentences.replace("\n", "")
        dev_article_sentences.append(sentences)
    
    write_iterate(train_article_sentences, "data/tuning/train.article")
    write_iterate(["Neglect."], "data/tuning/train.summary")
    write_iterate(dev_article_sentences, "data/tuning/valid.article")
    write_iterate(["Neglect."], "data/tuning/valid.summary")
    write_iterate(dev_article_sentences, "data/tuning/test.article")
    write_iterate(["Neglect."], "data/tuning/test.summary")
