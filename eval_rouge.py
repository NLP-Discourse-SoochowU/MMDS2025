from rouge_score import rouge_scorer
from app_cfg import *
import json
from util.file_util import *

def rouge_(pred_summaries, gold_summaries):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    r1_f, r2_f, rl_f = 0., 0., 0.
    items_all = len(pred_summaries)
    count_others = 0
    assert len(pred_summaries) == len(gold_summaries)
    for pred_summary, gold_summary in zip(pred_summaries, gold_summaries):
        score = scorer.score(pred_summary, gold_summary)
        r1_f += score["rouge1"][2]
        r2_f += score["rouge2"][2]
        rl_f += score["rougeL"][2]
    r1_f /= items_all
    r2_f /= items_all
    rl_f /= items_all
    return r1_f, r2_f, rl_f 

def report_rouge(summary_name, eval_thr, file_path="data/mds_clusters_gpt_event_dev.json", target="gpt4", skip_term=None):
    pred_summaries, gold_summaries = list(), list()
    with open(file_path, "r") as f:
        ac_list = json.load(f)
        for ac in ac_list:
            ac_sent_num = 0
            for doc_one in ac["documents"]:
                ac_sent_num += len(doc_one["sentences"])
            if ac_sent_num < eval_thr:
                continue
            summary = ac[summary_name]
            if "Despite<|im_start|>" in summary:
                summary = summary.split("Despite<|im_start|>")[0]
            gpt_summ = ac[target]
            if summary == "GPT: [OpenAI-Ban]" or (skip_term is not None and skip_term in summary):
                continue
            pred_summaries.append(summary)
            gold_summaries.append(gpt_summ)
    
    r1_f, r2_f, rl_f = rouge_(pred_summaries, gold_summaries)
    avg_value = (r1_f + r2_f + rl_f) / 3.
    print(f"Avg: {avg_value}, Rouge-1: {r1_f}, -2: {r2_f}, -L: {rl_f}")


if __name__ == '__main__':
    # ==================================
    # Ours
    # Test
    name_list_ = ["raw_seallm", "predicted-0.5-raw_seallm", "predicted-ent-len-0.8-raw_seallm", "baseline", "baseline-gpt4", "predicted-0.5", "predicted-ent-len-0.8", "llm_local", "predicted-ent-len-0.8-ctx", "llm_local_ctx"]
    for name in name_list_:
        print(name)
        report_rouge(name, file_path="data/mds_clusters_gpt_event.json", eval_thr=0, target="gpt4o")

    # ==================================
    # HIT
    # Test
    # name_list_ = ["raw_qwen", "predicted-0.9-raw_qwen", "predicted-ent-len-0.8-raw_qwen", "baseline", "predicted-0.4", "predicted-ent-len-0.3", "llm_local", "baseline-gpt35", "predicted-0.4-gpt35", "predicted-ent-len-0.3-gpt35", "baseline-gpt4"]
    # # name_list_ = [f"predicted-ent-len-{round(ratio_ * 0.1, 2)}-gpt35" for ratio_ in range(1, 11)]
    # for name in name_list_:
    #     print(name)
    #     report_rouge(name, file_path="data/mmds_test_hit_filtered.json", eval_thr=0)