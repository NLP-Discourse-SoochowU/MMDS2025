import spacy
from dateutil import parser
import re
from azure import gpt_response
import json
from tqdm import tqdm
from util.ml_sent_rep import split_sentences

nlp = spacy.load("en_core_web_sm")

def convert_to_formal(date_string):
    try:
        year_match = re.search(r'\b\d{4}\b', date_string)
        month_match = re.search(r'\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b', date_string, re.IGNORECASE)
        day_match = re.search(r'\b\d{1,2}\b', date_string)

        # Parse the date with default values for missing components
        parsed_date = parser.parse(date_string, default=None)
        
        # Determine which parts of the date are actually present
        has_year = bool(year_match)
        has_month = bool(month_match)
        has_day = bool(day_match)  # Avoid matching the year as day

        # Format based on available components
        if has_year and has_month and has_day:
            return parsed_date.strftime('%Y%m%d')  # Full date: YYYYMMDD
        elif has_year and has_month:
            return parsed_date.strftime('%Y%m')     # Year + Month: YYYYMM
        elif has_month and has_day:
            return parsed_date.strftime('%m%d')     # Month + Day: MMDD
        elif has_year:
            return parsed_date.strftime('%Y')       # Only Year: YYYY
        else:
            return "NA"
    except (ValueError, TypeError):
        return "NA"

def extract_entities(sentence):
    entity_data = {"named_entities": set(), "named_entities_date": set(), "numerics": set()}
    doc = nlp(sentence)
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "ORG", "GPE", "LOC"}:  # Named entities of interest
            entity_data["named_entities"].add((ent.text).lower())
        if ent.label_ in {"DATE"}:  # Named entities of interest
            date_formal = convert_to_formal((ent.text).lower())
            entity_data["named_entities_date"].add(date_formal)
        elif ent.label_ == "CARDINAL" or ent.label_ == "QUANTITY" or ent.label_ == "MONEY":
            entity_data["numerics"].add((ent.text).lower())  # Numerical values
    return entity_data

def evaluate_faithfulness(source_entities, summary_entities):
    # Calculate entity overlaps
    named_overlap = source_entities["named_entities"].intersection(summary_entities["named_entities"])
    date_overlap = source_entities["named_entities_date"].intersection(summary_entities["named_entities_date"])
    numeric_overlap = source_entities["numerics"].intersection(summary_entities["numerics"])
    
    # Evaluation
    faithfulness_score = {
        "named_entity_match": len(named_overlap) / max(len(summary_entities["named_entities"]), 1),
        "date_match": len(date_overlap) / max(len(summary_entities["named_entities_date"]), 1),
        "numeric_match": len(numeric_overlap) / max(len(summary_entities["numerics"]), 1)
    }
    return faithfulness_score

def eval_summ_faithfulness(summary_name, src_file="data/mds_clusters_gpt_event.json", trg_file="data/faith_eval.json", name_list=None, early_stop=None):
    # 1. prepare the entities for src sentences
    import httpcore
    setattr(httpcore, 'SyncHTTPTransport', 'AsyncHTTPProxy')
    from util.translate import google_trans, google_trans_hit

    with open(src_file, "r") as f:
        ac_list = json.load(f)
        ac_list_new = list()
        for ac_item in ac_list:
            documents = ac_item["documents"]
            sentences_all = []
            for doc_item in documents:
                sent_list = doc_item["sentences"]
                # translate to english
                for item in sent_list:
                    en_content = google_trans_hit(item["content"]) if "hit" in src_file else google_trans(item["content"])
                    sentences_all.append(en_content)
            # get src sent entities
            src_entities_all = list()
            for src_sent in tqdm(sentences_all):
                src_entities = extract_entities(src_sent)
                src_entities["named_entities"] = list(src_entities["named_entities"])
                src_entities["named_entities_date"] = list(src_entities["named_entities_date"])
                src_entities["numerics"] = list(src_entities["numerics"])
                if len(src_entities["named_entities"]) + len(src_entities["named_entities_date"]) + len(src_entities["numerics"]) > 0:
                    src_entities_all.append(src_entities)
            ac_item["sentence_entities"] = src_entities_all
            ac_list_new.append(ac_item.copy())

            with open(trg_file, "w") as f:
                json.dump(ac_list_new, f, indent=1)

    # 2. update trg_file for faith evaluation
    with open(src_file, "r") as f:
        ac_list = json.load(f)
    with open(trg_file, "r") as f:
        faith_list = json.load(f)
    faith_list_new = list()
    for faith_item, ac_item in zip(faith_list, ac_list):
        for name_one in name_list:
            faith_item[name_one] = ac_item[name_one]
        faith_list_new.append(faith_item.copy())
    with open(trg_file, "w") as f:
        json.dump(faith_list_new, f, indent=1)

    # 3. eval
    count_all_name, count_cor_name = 0., 0.
    ac_list = faith_list_new[:]

    if early_stop is not None:
        ac_list = [ac_list[early_stop]]
        print(ac_list[0][summary_name])

    for ac_item in ac_list:
        if ac_item[summary_name] == "GPT: [OpenAI-Ban]" or "<|endoftext|>" in ac_item[summary_name]:
            continue

        src_entities_all = ac_item["sentence_entities"]
        summary_sentences = split_sentences(ac_item[summary_name])
        for sum_sentence in summary_sentences:
            summ_entities = extract_entities(sum_sentence)
            name_flag, date_flag, num_flag = False, False, False
            if len(summ_entities["named_entities"]) > 0:
                name_flag = True
            if len(summ_entities["named_entities_date"]) > 0:
                date_flag = True
            if len(summ_entities["numerics"]) > 0:
                num_flag = True
            if name_flag or date_flag or num_flag:
                count_all_name += 1.
            else:
                continue

            # compare
            for src_entities in src_entities_all:
                src_entities["named_entities"] = set(src_entities["named_entities"])
                src_entities["named_entities_date"] = set(src_entities["named_entities_date"])
                src_entities["numerics"] = set(src_entities["numerics"])
                correct = 0
                faithfulness_result = evaluate_faithfulness(src_entities, summ_entities)
                if not name_flag or faithfulness_result["named_entity_match"] == 1:
                    correct += 1.
                if not date_flag or faithfulness_result["date_match"] == 1:
                    correct += 1.
                if not num_flag or faithfulness_result["numeric_match"] == 1:
                    correct += 1.
                if correct == 3.:
                    count_cor_name += 1.
                    break
    print("Entity-level faithfulness accuracy: ", count_cor_name / count_all_name)


if __name__ == '__main__':
    # ==================================
    # Ours
    name_list_ = ["raw_seallm", "predicted-0.5-raw_seallm", "predicted-ent-len-0.8-raw_seallm", "baseline", "baseline-gpt4", "predicted-0.5", "predicted-ent-len-0.8", "llm_local", "llm_local_sg", "llm_local_ctx", "predicted-ent-len-0.8-ctx"]
    # for name in name_list_:
    #     print(name)
    #     eval_summ_faithfulness(name, name_list=name_list_)
    eval_summ_faithfulness("llm_local_ctx", name_list=name_list_)
    eval_summ_faithfulness("predicted-ent-len-0.8-ctx", name_list=name_list_)
    
    # Analysis
    # compare_idx = 0
    # while True:
    #     try:
    #         eval_summ_faithfulness("llm_local", name_list=name_list_, early_stop=compare_idx)
    #         eval_summ_faithfulness("baseline-gpt4", name_list=name_list_, early_stop=compare_idx)
    #     except:
    #         pass
    #     compare_idx += 1
    #     input()

    # ==================================
    # HIT
    # name_list_ = ["raw_qwen", "predicted-0.9-raw_qwen", "predicted-ent-len-0.8-raw_qwen", "baseline", "predicted-0.4", "predicted-ent-len-0.3", "llm_local", "baseline-gpt35", "predicted-0.4-gpt35", "predicted-ent-len-0.3-gpt35", "baseline-gpt4"]
    # # name_list_ = [f"predicted-ent-len-{round(ratio_ * 0.1, 2)}-gpt35" for ratio_ in range(1, 11)]
    # for name in name_list_:
    #     print(name)
    #     eval_summ_faithfulness(name, src_file="data/mmds_test_hit_filtered.json", trg_file="data/faith_eval_hit_filtered.json", name_list=name_list_)
