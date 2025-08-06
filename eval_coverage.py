import json
from sentence_transformers import SentenceTransformer
from util.ml_sent_rep import split_sentences
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from run_mmds import rDoc
import csv
from tqdm import tqdm
import pandas as pd
from util.nlp_util import entropy, compression_ratio
from util.file_util import save_data, load_data
from azure import gpt_response

model = SentenceTransformer('ml_models/sent_bert')  # a cross-lingual version
model.eval()

def coverage_eval(threshold=0.6, summary_goal="gpt4o", summary_goal2=None, eval_thr=90, test_data_path=None, skip_term=None):
    coverage_rate = 0.
    a, b, c = 0., 0., 0.
    with open(test_data_path, "r") as f:
        instances = json.load(f)
        for ac_id, ac in enumerate(instances):
            ac_manual_events = []
            docs = ac["documents"]
            ac_sent_num = 0
            for doc_ in docs:
                related_events = doc_["gpt_event_labels"]

                main_events = []
                main_event = "Unknown"
                doc_sentences = doc_["sentences"]
                ac_sent_num += len(doc_sentences)
                for item in doc_sentences:
                    content_ = item['content'].replace("\n", " ")
                    sentence_id = f"##Sentence-{item['SID']}##"
                    sentence_id1 = f"{item['SID']},"
                    sentence_id2 = f",{item['SID']}"
                    if (sentence_id in related_events) or (sentence_id1 in related_events) or (sentence_id2 in related_events):
                        main_events.append(content_)
                    if item["event_annotation"] == "main":
                        main_event = content_
                if main_event != "Unknown":
                    main_events = [main_event] + main_events
                ac_manual_events += main_events  # Maybe in different languages
            
            if ac_sent_num < eval_thr:
                continue

            if ac[summary_goal] == "GPT: [OpenAI-Ban]" or (skip_term is not None and skip_term in ac[summary_goal]):
                continue

            # summary sentences hit?
            ac_summary_sentences = split_sentences(ac[summary_goal])
            summ_sent_embeddings = model.encode(ac_summary_sentences)
            event_embeddings = model.encode(ac_manual_events)
            similarity_matrix = cosine_similarity(summ_sent_embeddings, event_embeddings)
            max_values = np.max(similarity_matrix, axis=1)  # Shape (1, 100)
            count_beyond = np.sum(max_values >= threshold)
            a += count_beyond
            b += len(ac_summary_sentences)
            c += len(ac_manual_events)
    p_ = a / b
    r_ = a / c
    f_ = 2 * p_ * r_ / (p_ + r_)
    print("P: ", p_, "R: ", r_, "F1: ", f_)
    return f_

def coverage_eval_ana(threshold=0.6, summary_goal="gpt4o", eval_thr=90, test_data_path=None, skip_term=None, ac_number=None):
    coverage_rate = 0.
    a, b, c = 0., 0., 0.
    count_ac = 0
    our_judges = []
    with open(test_data_path, "r") as f:
        instances = json.load(f)
        results_all = []
        for ac_id, ac in enumerate(instances):
            ac_manual_events = []
            docs = ac["documents"]
            ac_sent_num = 0
            for doc_ in docs:
                related_events = doc_["gpt_event_labels"]

                main_events = []
                main_event = "Unknown"
                doc_sentences = doc_["sentences"]
                ac_sent_num += len(doc_sentences)
                for item in doc_sentences:
                    content_ = item['content'].replace("\n", " ")
                    sentence_id = f"##Sentence-{item['SID']}##"
                    sentence_id1 = f"{item['SID']},"
                    sentence_id2 = f",{item['SID']}"
                    if (sentence_id in related_events) or (sentence_id1 in related_events) or (sentence_id2 in related_events):
                        main_events.append(content_)
                    if item["event_annotation"] == "main":
                        main_event = content_
                if main_event != "Unknown":
                    main_events = [main_event] + main_events
                ac_manual_events += main_events 
            
            if ac_sent_num < eval_thr:
                continue
            if ac[summary_goal] == "GPT: [OpenAI-Ban]" or (skip_term is not None and skip_term in ac[summary_goal]):
                continue

            # summary sentences hit?
            ac_summary_sentences = split_sentences(ac[summary_goal])
            summ_sent_embeddings = model.encode(ac_summary_sentences)
            event_embeddings = model.encode(ac_manual_events)
            similarity_matrix = cosine_similarity(summ_sent_embeddings, event_embeddings)
            max_values = list(np.max(similarity_matrix, axis=1))  # Shape (1, 100)
            max_indexes = [list(row).index(max(row)) for row in similarity_matrix]
            our_judges += [sim_item >= threshold for sim_item in max_values]
            results_ = [(summ_item, ac_manual_events[idx_], sim_item >= threshold) for summ_item, idx_, sim_item in zip(ac_summary_sentences, max_indexes, max_values)]
            results_all += results_[:]
            c += len(ac_manual_events)
            count_ac += 1
            if ac_number is not None and count_ac >= ac_number:
                break
    results_all = (results_all, c)
    
    if "hit" in test_data_path:
        save_data(results_all, "data/cov_hit_judge.pkl")
    else:
        save_data(results_all, "data/cov_our_judge.pkl")

def gpt_judge(use_hit=False):
    source_path = "data/cov_hit_judge.pkl" if use_hit else "data/cov_our_judge.pkl"
    source_items, events_all = load_data(source_path)
    target_items = list()
    for item in tqdm(source_items):
        sum_it, gold_it, our_judge, _ = item
        ac_ask = f"The following are two events:\nEvent 1: {sum_it}\nEvent 2: {gold_it}\nJudge if the two events are the same or not, simply return True or False as answer."
        response = gpt_response(ac_ask, gpt_type="gpt4").lower()
        response = "true" in response
        target_items.append((sum_it, gold_it, our_judge, response))
    save_data((target_items, events_all), source_path)

def gpt_compare(use_hit=False, human_feedback=False):
    source_path = "data/cov_hit_judge.pkl" if use_hit else "data/cov_our_judge.pkl"
    source_items, events_all = load_data(source_path)
    print("Total number of judgements: ", len(source_items))
    
    all_pairs = len(source_items)
    human_items = []
    our_corr_pairs, gpt_corr_pairs = 0., 0.
    for item in tqdm(source_items):
        sum_it, gold_it, our_judge, gpt4_judge = item
        if our_judge == gpt4_judge:
            our_corr_pairs += 1.
            gpt_corr_pairs += 1.
            human_ann = our_judge
        elif human_feedback and our_judge == True:
            our_corr_pairs += 1.
            human_ann = True
        elif human_feedback:
            print("S1: ", sum_it)
            print("S2: ", gold_it)
            print(our_judge, gpt4_judge)
            human_judge = input("Agree with our metric or GPT? (0 or 1)")
            if human_judge == "0":
                our_corr_pairs += 1.
                human_ann = our_judge
            else:
                gpt_corr_pairs += 1.
                human_ann = gpt4_judge
        else:
            human_ann = None
        human_items.append((sum_it, gold_it, our_judge, gpt4_judge, human_ann))
    human_agreement_our_metric = our_corr_pairs / all_pairs
    print("Agreement of ours: ", human_agreement_our_metric)
    human_agreement_gpt = gpt_corr_pairs / all_pairs
    print("Agreement of GPT4: ", human_agreement_gpt)
    save_data(human_items, source_path + ".human")


def compare_cov_performance(use_hit=False):
    source_path = "data/cov_hit_judge.pkl" if use_hit else "data/cov_our_judge.pkl"
    source_items, events_all = load_data(source_path)


def draw_cluster_entropy(eval_thr=90, draw_entropy=True):
    target = "output" if draw_entropy else "output_com"
    pred_summaries, gold_summaries = list(), list()
    value_group = [[] for _ in range(10)]
    with open("data/mds_clusters_gpt_event.json", "r") as f:
        ac_list = json.load(f)
        ac_list_new = list()
        ac_sent_num_distribute = list()
        for ac in tqdm(ac_list):
            docs = ac["documents"]
            new_ac = list()
            ac_sent_num = 0
            for doc_one in docs:
                sents = [item["content"].strip().replace("\n", " ") for item in doc_one["sentences"]]
                article_id = doc_one["ID"]
                org_title = doc_one["title"]
                new_ac.append((article_id, sents, org_title))
                ac_sent_num += len(sents)
            if ac_sent_num >= eval_thr:
                ac_sent_num_distribute.append(ac_sent_num)
            else:
                continue

            # form new doc
            group_min = 100000000
            group_max = -100000000
            group_values = []
            for doc_ratio in np.arange(0.1, 1.1, 0.1):
                doc_ratio = round(doc_ratio, 2)
                if doc_ratio == 1.0:
                    new_doc = []
                    for item_doc in new_ac:
                        _, doc_sents, _ = item_doc
                        new_doc += doc_sents
                else:
                    new_doc, _ = rDoc(new_ac, doc_ratio_type=2, doc_ratio=doc_ratio)  # fixed rate
                new_doc_ = " ".join(new_doc)

                g_value = entropy(new_doc_) if draw_entropy else compression_ratio(new_doc_, get_com=True)[0]

                group_min = min(group_min, g_value)
                group_max = max(group_max, g_value)
                group_values.append(g_value)

            for group_idx, group_item in enumerate(group_values):
                add_vale = [round(0.0001 + (1 - 0.0002) * (group_item - group_min) / (group_max - group_min), 2)]  # if draw_entropy else group_item
                value_group[group_idx] = value_group[group_idx] + add_vale
            
            # Writing to a CSV file
            with open(f'data/test/{target}.csv', mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(value_group)

        # write the first line
        with open(f'data/test/{target}.csv', mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            existing_rows = list(reader)
        updated_rows = [ac_sent_num_distribute] + existing_rows
        with open(f'data/test/{target}.csv', mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(updated_rows)

if __name__ == '__main__':
    # dev
    # for doc_ratio in range(1, 11):
    #     doc_ratio = round(doc_ratio * 0.1, 2)
        # coverage_score = coverage_eval(0.75, f"predicted-{doc_ratio}", eval_thr=0, test_data_path="data/mds_clusters_gpt_event_dev.json")
        # coverage_score = coverage_eval(0.75, f"predicted-ent-len-{doc_ratio}", eval_thr=0, test_data_path="data/mds_clusters_gpt_event_dev.json")

    # test
    """
    ==================================
    Baseline
    Summary_goal:  baseline F1:  0.28064903846153844

    Highest information entropy
    Summary_goal:  predicted-0.5 F1:  0.29504303947758975

    Balanced volume & information entropy
    Summary_goal:  predicted-ent-len-0.3 F1:  0.2958790394307738

    Balanced volume & information entropy + Local_TQA
    Summary_goal:  llm_local F1:  0.31892697466467956

    GPT4
    Summary_goal:  gpt4 F1:  0.14687100893997446
    """
    # Ours
    # Dev
    # for ratio_ in range(1, 11):
    #     doc_hype = round(ratio_ * 0.1, 2)
    #     key_name = f"predicted-{doc_hype}"
    #     key_name = f"predicted-ent-len-{doc_hype}"
    #     coverage_eval(0.75, key_name, eval_thr=0, test_data_path="data/mds_clusters_gpt_event_dev.json")

    # # Test
    # name_list_ = ["raw_seallm", "predicted-0.5-raw_seallm", "predicted-ent-len-0.8-raw_seallm", "baseline", "predicted-0.5", "predicted-ent-len-0.8", "llm_local", "baseline-gpt4"]
    # for name in name_list_:
    #     print(name)
    #     coverage_eval(0.75, name, eval_thr=0, test_data_path="data/mds_clusters_gpt_event.json")
    coverage_eval(0.75, "llm_local_ctx", eval_thr=0, test_data_path="data/mds_clusters_gpt_event.json")
    # coverage_eval(0.75, "predicted-ent-len-0.8-ctx", eval_thr=0, test_data_path="data/mds_clusters_gpt_event.json")

    # Hit
    # Dev
    # for ratio_ in range(1, 11):
    #     doc_hype = round(ratio_ * 0.1, 2)
    #     # key_name = f"predicted-{doc_hype}"
    #     key_name = f"predicted-ent-len-{doc_hype}"
    #     # key_name = key_name + "-raw_qwen"
    #     coverage_eval(0.6, key_name, eval_thr=0, test_data_path="data/mmds_valid_hit_filtered.json")
    
    # Test
    # name_list_ = ["raw_qwen", "predicted-0.9-raw_qwen", "predicted-ent-len-0.8-raw_qwen", "baseline", "predicted-0.4", "predicted-ent-len-0.3", "llm_local", "baseline-gpt35", "predicted-0.4-gpt35", "predicted-ent-len-0.3-gpt35", "baseline-gpt4"]
    # # name_list_ = [f"predicted-ent-len-{round(ratio_ * 0.1, 2)}-raw_qwen" for ratio_ in range(1, 11)]
    # for name in name_list_:
    #     print(name)
    #     coverage_eval(0.6, name, eval_thr=0, test_data_path="data/mmds_test_hit_filtered.json")

    # Translation results
    # coverage_score = coverage_eval(0.75, "predicted-ent-len-en", eval_thr=0, test_data_path="data/mds_clusters_gpt_event_en.json")

    # =================================================
    # Analysis
    # Figure 1
    # draw_cluster_entropy(eval_thr=70, draw_entropy=True)  
    # draw_cluster_entropy(eval_thr=70, draw_entropy=False)  
    
    # Figure Ana
    # for ratio_ in range(1, 11):  
    #     doc_hype = round(ratio_ * 0.1, 2)
    #     key_name = f"predicted-{doc_hype}"
    #     coverage_eval(0.75, key_name, eval_thr=0, test_data_path="data/mds_clusters_gpt_event_dev.json")
    
    # coverage_score = coverage_eval(0.75, "predicted-ent-len-0.8", summary_goal2="baseline-gpt4", eval_thr=0, test_data_path="data/mds_clusters_gpt_event.json")

    # save the results for GPT4o estimation
    # coverage_eval_ana(0.75, "predicted-ent-len-0.8", eval_thr=0, test_data_path="data/mds_clusters_gpt_event_dev.json", ac_number=30)
    # gpt_judge(use_hit=False)
    # gpt_compare(use_hit=False)
    # gpt_compare(use_hit=False, human_feedback=True)
    
    # not run for HIT dataset
    # coverage_eval_ana(0.75, "predicted-ent-len-0.6", eval_thr=0, test_data_path="data/mmds_valid_hit_filtered.json", ac_number=30)
    # gpt_judge(use_hit=True)
    # gpt_compare(use_hit=True)
