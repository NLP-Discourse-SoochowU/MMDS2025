import argparse
import torch
from rouge_score import rouge_scorer
from cluster_models.vers1.trainer import Trainer as Trainer1
from util.ml_sent_rep import chat_multiturn_seq_format_v2, parse_seallm_v2
from transformers import AutoModel, XLMRobertaTokenizer, AutoModelForSeq2SeqLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, GenerationConfig, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from app_cfg import *
import json
from sklearn import metrics
from util.file_util import *
from tqdm import tqdm
import logging
from train_pairing import MultiTaskModel
import time
import torch.nn.functional as F
from InstructorEmbedding import INSTRUCTOR
from util.ml_sent_rep import split_sentences
from util.nlp_util import entropy, compression_ratio
from azure import gpt_response
from gpt import chat_gpt
import random
import math
from run_me import get_me
torch.manual_seed(7)

logger = logging.getLogger(__name__)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_lora = None
llm_tokenizer = None
generation_config = None
hit_summ_test = False

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

def load_model(load_tqa=False, load_lora=True):
    print("Loading models...")
    global model_lora, llm_tokenizer, generation_config
    if hit_summ_test:
        base_model_path = "ml_models/Qwen2.5-7B-Instruct"
        mmds_lora = "ml_models/checkpoints/qwen_mmds_filter/checkpoint-1000/"
        mmds_lora_local = "ml_models/checkpoints/qwen_mmds_qa_filter/checkpoint-1000/"
        tuned_lora = mmds_lora_local if load_tqa else mmds_lora
        llm_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        llm_tokenizer.pad_token = llm_tokenizer.eos_token  # Important for Qwen models
        if load_lora:
            model_ = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16, quantization_config=bnb_config, device_map=f"cuda")
            model_.config.use_cache = False
            model_lora = PeftModel.from_pretrained(model_, tuned_lora, torch_dtype=torch.bfloat16)
            model_lora = model_lora.to(f"cuda")
        else:
            model_lora = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16, quantization_config=bnb_config, device_map=f"cuda")
            model_lora.config.use_cache = False
        model_lora.eval()
    else:
        base_model_path = "ml_models/seallm_7b/7B_V2"
        mmds_lora = "ml_models/checkpoints/mmds_ctx/checkpoint-1200/"  
        mmds_lora_local = "ml_models/checkpoints/mmds_qa_ctx/checkpoint-1200/"  # local knowledge through Tempora QA + limited context
        tuned_lora = mmds_lora_local if load_tqa else mmds_lora
        llm_tokenizer = AutoTokenizer.from_pretrained(base_model_path, padding_side='left')
        llm_tokenizer.add_special_tokens = False
        if load_lora:
            print(tuned_lora)
            model_ = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16, device_map=f"cuda")  # quantization_config=bnb_config, 
            model_.config.use_cache = False
            model_lora = PeftModel.from_pretrained(model_, tuned_lora, torch_dtype=torch.bfloat16)
            model_lora = model_lora.to(f"cuda")
        else:
            model_lora = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16, device_map=f"cuda")  # quantization_config=bnb_config, 
            model_lora.config.use_cache = False
        model_lora.eval()
        generation_config = GenerationConfig(do_sample=True, temperature=0.5, top_p=1.0, top_k=5, num_beams=2, eos_token_id=llm_tokenizer.eos_token_id, pad_token_id=llm_tokenizer.eos_token_id, pad_token=llm_tokenizer.pad_token)
    torch.cuda.empty_cache()

if use_ac_me:
    me_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
    me_model = MultiTaskModel('xlm-roberta-large')
    me_model.load_state_dict(torch.load(f'ml_models/me_models/multitask_model_epoch4.pt', map_location=device))  # should select models on dev
    me_model.to(device)
    me_model.eval()

if use_ac_t3:
    t3_tokenizer = AutoTokenizer.from_pretrained('ml_models/sent_bert', padding_side='left')
    t3_model = AutoModel.from_pretrained('ml_models/sent_bert')
    t3_model.to(device)
    t3_model.eval()

def seallm_style(ask):
    return f"<|im_start|>system \nYou are a helpful assistant.</s><|im_start|>user \nHello world</s><|im_start|>assistant \nHi there, how can I help you today?</s><|im_start|>user \n{ask}</s>"

def qwen_style(context, instruct):
    return f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>context\n{context}<|im_end|>\n<|im_start|>instruction\n{instruct}<|im_end|>\n<|im_start|>response\n"

def extract_sentence(sentences):
    tokenized_inputs = me_tokenizer(sentences, padding=False, truncation=False, max_length=512)
    max_length = min(512, max(len(tokens) for tokens in tokenized_inputs['input_ids']))

    encoded_inputs = me_tokenizer(sentences, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
    input_ids = encoded_inputs['input_ids'].to(device)
    attention_mask = encoded_inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = me_model(input_ids=input_ids, attention_mask=attention_mask, task='sentence_extraction')
    
    predicted_sentence_id = torch.argmax(outputs).item()
    return sentences[predicted_sentence_id], predicted_sentence_id

def pipeline_rouge(summ_p, summ_g):
    eval_model = INSTRUCTOR("hkunlp/instructor-large").to(device)
    instruction = "Represent the statement:"
    pred_true = 0.
    rouge_p_summaries, rouge_g_summaries = [], []
    for sent_pred in tqdm(summ_p):
        gold_sentences, pred_sentences = list(), list()
        sent_pred = sent_pred.lower()
        for sent_gold in summ_g:
            sent_gold = sent_gold.lower()
            gold_sentences.append(sent_gold)
            pred_sentences.append(sent_pred)

        with torch.no_grad():
            gold_sentences_ = [[instruction, sent_item] for sent_item in gold_sentences]
            gold_embeddings = eval_model.encode(gold_sentences_, batch_size=64, show_progress_bar=False, convert_to_tensor=True)
            gold_embeddings = gold_embeddings.to(device)

            pred_sentences_ = [[instruction, sent_item] for sent_item in pred_sentences]
            pred_embeddings = eval_model.encode(pred_sentences_, batch_size=64, show_progress_bar=False, convert_to_tensor=True)
            pred_embeddings = pred_embeddings.to(device)

            # Normalize the matrices
            cosine_sim = F.cosine_similarity(gold_embeddings, pred_embeddings, dim=1).cpu().numpy().tolist()
            max_score = max(cosine_sim)
            idx_ = cosine_sim.index(max_score)
            if max_score >= 0.95:
                pred_true += 1.
                rouge_p_summaries.append(sent_pred)
                rouge_g_summaries.append(gold_sentences[idx_])

    r1_f, r2_f, rl_f = rouge_(rouge_p_summaries, rouge_g_summaries)
    hit_p = pred_true / len(summ_p)
    hit_r = pred_true / len(summ_g)
    hit_score = 2 * hit_p * hit_r / (hit_p + hit_r)

    o_r1_f, o_r2_f, o_rl_f = r1_f * hit_score, r2_f * hit_score, rl_f * hit_score

    print("Hit Rate: {}-P, {}-R, {}-F1".format(hit_p, hit_r, hit_score))
    print("Rouge: {}-R1, {}-R2, {}-RL".format(r1_f, r2_f, rl_f))
    print("Overall: {}-R1, {}-R2, {}-RL".format(o_r1_f, o_r2_f, o_rl_f))
    score_string = f"Hit Rate: {hit_p}-P, {hit_r}-R, {hit_score}-F1"
    write_append(score_string, "data/pipeline_logs.txt")

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

def clustering(data_set, cversion=1, clustering_thr_=[0.54, 0.95, 0.85, 200], background_info=None, wokw=False):
    comment_num = len(data_set)
    trainer = Trainer1(data_set, comment_num, clustering_thr_, background_info)
    clusters, pre_ucc_labels = trainer.train(logger, wokw)
    return clusters, pre_ucc_labels

def t3sentences_ml(sentences, chunk_size=512):
    """ Chunking the article for K blocks, then select sentences closest to the three.
        This process all based on ML data.
    """
    article = " ".join(sentences)
    article_tokens = article.split()
    chunks = [" ".join(article_tokens[i:i + chunk_size]) for i in range(0, len(article_tokens), chunk_size)]
    if len(chunks) > 1:
        chunks = chunks[:-1]  # remove the tail, could be noisy
    
    if len(sentences) <= 3:
        context_str = " ".join(sentences)
        return context_str

    # Tokenize sentences & chunks
    sentence_embeddings_norm = sentences_rep(sentences)
    chunk_embeddings_norm = sentences_rep(chunks)

    cosine_similarity = torch.mm(sentence_embeddings_norm, chunk_embeddings_norm.t())

    cosine_similarity = cosine_similarity.sum(dim=-1).squeeze()
    top_indices = torch.topk(cosine_similarity, 3).indices

    context_str = " ".join([sentences[idx] for idx in top_indices])
    return context_str

def sentences_rep(texts):
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    encoded_texts = t3_tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to("cuda")
    with torch.no_grad():
        model_output_texts = t3_model(**encoded_texts)
    text_embeddings = mean_pooling(model_output_texts, encoded_texts['attention_mask'])
    text_embeddings_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
    return text_embeddings_norm

def art_cluster(args_, articles_sents, titles, art_ids, similar_max_min, art_event=None, art_trigger=None, art_who=None, art_where=None, art_when=None, art_outcome=None):
    # 1. main event extraction for each article
    start_time = time.time()
    me_list = list()
    art_id_list = list()
    for sentences, tt, art_id, art_event_, art_trigger_, art_who_, art_where_, art_when_, art_outcome_ in zip(articles_sents, titles, art_ids, art_event, art_trigger, art_who, art_where, art_when, art_outcome):
        art_id_list.append(art_id)
        pred_me = []
        if args_.use_ac_tt:
            pred_me.append(tt)
        if args_.use_ac_t3:
            pred_me_plus = t3sentences_ml(sentences)
            pred_me.append(pred_me_plus)
        if args_.use_ac_1st:
            pred_me_plus = "" if len(sentences) == 0 else " ".join(sentences[:3])
            pred_me.append(pred_me_plus)
        if args_.use_ac_me:
            if len(sentences) == 0:
                pred_me_plus = ""
            elif len(sentences) == 1:
                pred_me_plus = sentences[0]
            else:
                pred_me_plus, _ = extract_sentence(sentences)
            pred_me.append(pred_me_plus)
        event_str = ""
        if args_.use_event:
            event_str = event_str + art_event_
        if args_.use_trigger:
            event_str = event_str + "; " + art_trigger_
        if args_.use_who:
            event_str = event_str + "; " + art_who_
        if args_.use_where:
            event_str = event_str + "; " + art_where_
        if args_.use_when:
            event_str = event_str + "; " + art_when_
        if args_.use_outcome:
            event_str = event_str + "; " + art_outcome_
        if event_str != "":
            pred_me.append(event_str)
        # event attributions
        me_list.append(pred_me)
    runtime = time.time() - start_time

    # 2. clustering
    start_time = time.time()
    sentences_all = [(me_item, id_item) for me_item, id_item in zip(me_list, art_id_list)]
    clusters, _ = clustering(sentences_all, cversion=args_.cversion, clustering_thr_=similar_max_min, wokw=True)
    art_id_clusters = [item[1] for item in clusters]
    runtime = time.time() - start_time
    return art_id_clusters

def rDoc(tune_ac, clustering_thr=[0.54, 0.95, 0.85, 200], doc_ratio=0.5, doc_ratio_type=3, hp1=0.5, hp2=0.5):
    background = [item[2] for item in tune_ac[:3]] 
    ac_article_ids = [item[0] for item in tune_ac]
    sentences_all = []
    publish_info = [] if len(tune_ac[0]) <= 3 else [item[3] for item in tune_ac]

    periods = []
    period_begin = 0
    for item in tune_ac:
        sentences_all += item[1]
        sent_count = len(item[1])
        periods.append((period_begin, period_begin + sent_count - 1))
        period_begin += sent_count
    SIDs = range(0, len(sentences_all))

    # cluster
    sentences_all = [(None, sent_item, sent_id) for sent_item, sent_id in zip(sentences_all, SIDs)]
    sentences_all_str = " ".join([sent_item[1] for sent_item in sentences_all])
    clusters, _ = clustering(sentences_all, cversion=1, clustering_thr_=clustering_thr)
    other_cluster_size = len(clusters[-1])
    clusters = clusters[:-1]  # In this case, we do not consider the other sentence cluster
    
    new_doc, new_doc_sids = list(), list()
    sentences_number = len(sentences_all) - other_cluster_size

    assert doc_ratio_type != 1

    if sentences_number > 0:
        if doc_ratio_type == 2:
            sample_sentences_num = max(1, sentences_number * doc_ratio)
            for cluster in clusters:
                sentences, _, sentences_ids = cluster
                K = max(int(sample_sentences_num * len(sentences) / sentences_number), 1)
                new_doc += sentences[:K]
                new_doc_sids += sentences_ids[:K]
                if len(new_doc) >= sample_sentences_num:
                    break
        else:
            group_values = []
            group_min = 100000000
            group_max = -100000000
            group_vol_values = []
            group_vol_min = 100000000
            group_vol_max = -100000000
            group_docs = []
            f_value_max = -100000000
            final_doc_ratio = 0
            for doc_ratio in range(1, 11):
                new_doc_, new_doc_sids_ = [], []
                doc_ratio = round(doc_ratio * 0.1, 2)
                sample_sentences_num = max(1, sentences_number * doc_ratio)

                for cluster in clusters:
                    sentences, _, sentences_ids = cluster
                    K = max(int(sample_sentences_num * len(sentences) / sentences_number), 1)
                    new_doc_ += sentences[:K]
                    new_doc_sids_ += sentences_ids[:K]
                    if len(new_doc_) >= sample_sentences_num:
                        break
                group_docs.append((new_doc_, new_doc_sids_))

                # get f_value
                new_doc_str = " ".join(new_doc_)
                doc_entropy = entropy(new_doc_str)
                group_values.append(doc_entropy)

                doc_volume = compression_ratio(new_doc_str, get_com=True)[0]
                group_vol_min = min(group_vol_min, doc_volume)
                group_vol_max = max(group_vol_max, doc_volume)
                group_vol_values.append(doc_volume)

            for ent_val, vol_val, new_doc_info in zip(group_values, group_vol_values, group_docs):
                ent_temp = ent_val
                vol_temp = 0.0001 + (1 - 0.0002) * (vol_val - group_vol_min) / (group_vol_max - group_vol_min)
                f_value = hp1 * ent_temp + vol_temp 
                if f_value >= f_value_max:
                    new_doc_, new_doc_sids_ = new_doc_info
                    f_value_max = f_value
                    new_doc = new_doc_[:]
                    new_doc_sids = new_doc_sids_[:]
        # rank the sentences according to sent IDs, while for method we don't need ranking them, let the model to focus on each piece
        ranked_sentences = dict()
        for sent_, sent_id in zip(new_doc, new_doc_sids):
            ranked_sentences[sent_id] = sent_

        ranked_sent_ids = sorted(ranked_sentences.keys())
        new_doc = []
        if len(periods) > 0:
            current_period = periods.pop(0)
        doc_id = 0
        for idx in ranked_sent_ids:
            if current_period[0] <= idx <= current_period[1]:
                if doc_id != 0:
                    new_doc.append(f"##End of Article {doc_id}##\n")
                doc_id += 1
                if len(publish_info) > 0:
                    publish_date = publish_info.pop(0)
                else:
                    publish_date = "Unknown"
                art_description = f"##Article {doc_id} published on {publish_date}##" if publish_date != "Unknown" else f"##Article {doc_id}"
                new_doc.append(art_description)
                current_period = periods.pop(0) if len(periods) > 0 else (-100, -10)
            new_doc.append(ranked_sentences[idx])
        new_doc.append(f"##End of Article {doc_id}##\n")
    else:
        new_doc = ""
        ranked_sent_ids = []
    return new_doc, ranked_sent_ids

def summarize_rdoc(sentences, total_token_max=2048, gpt_use=False):
    sentences_str = []
    sentence_id = 1
    for item in sentences:
        if not item.startswith("##Article ") and not item.startswith("##End of Article "):
            sentences_str.append(f"Sentence {sentence_id}: " + item)
            sentence_id += 1
        else:
            sentences_str.append(item)
    sentences_to_cos = sentences_str[:]

    if gpt_use:
        pass_flag_ = False
        while not pass_flag_:
            try:
                if hit_summ_test:
                    ac_ask = "The following are sentences from articles in multiple languages, each sentence is attached with a unique sentence ID. Please identify events among these sentences and summarize the events into a fluent paragraph. The articles are:\n" + "\n".join(sentences_to_cos) + "\nSummarize the articles in English and ensure the summary is faithful, concise with a moderate length, and covers the main points. Ensure all proper nouns are fully presented, including person names, organization names, locations, etc. If the event has a clear occurrence time in the original text, please formally reflect the timestamp in the summary. Return the summary as a piece of text."
                    response = chat_gpt(ac_ask)
                else:
                    ac_ask = "The following are sentences from articles in languages like English, Chinese, or Malay, each sentence is attached with a unique sentence ID. Please identify events among these sentences and summarize the events into a fluent paragraph. The articles are:\n" + "\n".join(sentences_to_cos) + "\nSummarize the articles in English and ensure the summary is faithful, concise with a moderate length, and covers the main points. Ensure all proper nouns are fully presented, including person names, organization names, locations, etc. If the event has a clear occurrence time in the original text, please formally reflect the timestamp in the summary. Return the summary as a piece of text."
                    response = gpt_response(ac_ask, gpt_type="gpt4")
                pass_flag_ = True
            except:
                print(len(sentences_to_cos))
                sentences_to_cos.pop(-1)
        return response

    pass_flag = False
    prompt_made = None
    while not pass_flag:
        if hit_summ_test:
            context = "The following are sentences from articles in multiple languages, each sentence is attached with a sentence ID. The sentences are:\n" + "\n".join(sentences_to_cos)
            instruct = "Please identify events among these sentences and summarize the events into a fluent paragraph. If the event has a timestamp in source texts, please clearly reveal timestamps in formal style in the summarization. Return the summary in English and ensure the summary is faithful, concise with a moderate length, and covers the main points. Ensure that all proper nouns are fully presented, including person names, organization names, locations, and dates."
            prompt_made = qwen_style(context, instruct)
        else:
            ac_ask = "The following are sentences from articles in multiple languages, each sentence is attached with a sentence ID. Please identify events among these sentences and summarize the events into a fluent paragraph. If the event has a timestamp in source texts, please clearly reveal timestamps in formal style in the summarization. The sentences are:\n" + "\n".join(sentences_to_cos) + "\nReturn the summary in English and ensure the summary is faithful, concise with a moderate length, and covers the main points. Ensure that all proper nouns are fully presented, including person names, organization names, locations, and dates. Summarize using only the information from the input texts. Do not assume or infer facts not mentioned explicitly."
            prompt_made = seallm_style(ac_ask)
        sentence_tokens = llm_tokenizer.encode(prompt_made, add_special_tokens=False)
        count_tokens = len(sentence_tokens)
        if count_tokens <= total_token_max:
            pass_flag = True
        else:
            sentences_to_cos.pop(-1)

    if hit_summ_test:
        generation_params = {
            "max_new_tokens": 1000,     # Adjust for desired length of response
            "temperature": 0.7,         # Controls randomness, lower values more focused
            "top_p": 0.8,               # Nucleus sampling, higher values more diverse
            "top_k": 3,                 # Top-k sampling, integer for top-k words
            "repetition_penalty": 1.2,  # Discourages repetition
            "do_sample": True,          # Use sampling instead of greedy search
            "num_beams": 1,             # Number of beams for beam search (set to 1 for sampling or greedy)
            "length_penalty": 1.0       # Keep this at 1 for now.
        }
        cc_ids = llm_tokenizer(prompt_made, return_tensors="pt", max_length=2048, truncation=True).to("cuda")
        with torch.no_grad():
            generation_output = model_lora.generate(**cc_ids, **generation_params)
        response = llm_tokenizer.decode(generation_output[0])
        # input(response)
        response = response.split("<|im_start|>response")[-1].replace("<|im_end|>", "").replace("\n", "").strip()
        # input(response)
    else:
        inputs = llm_tokenizer(prompt_made, padding=True, return_tensors="pt", max_length=total_token_max, truncation=True)
        cc_ids = inputs["input_ids"].to("cuda")
        with torch.no_grad():
            generation_output = model_lora.generate(input_ids=cc_ids, generation_config=generation_config, return_dict_in_generate=True, output_scores=True, max_new_tokens=1000)
        s = generation_output.sequences[0]
        response = llm_tokenizer.decode(s).strip()
        # response = response.split("Ensure the summary be ended with 'END'")[1].replace("</s>", "").strip()
        response = response.split("Summarize using only the information from the input texts. Do not assume or infer facts not mentioned explicitly.")[1].replace("</s>", "").strip()
        
        response = response.strip()
    return response

def pipe_eval(args_):
    pred_summaries = list()
    with open("data/mds_ac_test_final.json", "r") as f:
        ac_test = json.load(f)
        arts = ac_test["articles"]
    arts_sentences, titles, art_ids, art_event, art_trigger, art_who, art_where, art_when, art_outcome = [], [], [], [], [], [], [], [], []
    for art_one in arts:
        tt_one = art_one["title"]
        id_one = art_one["ID"]
        sentences_one = [item["content"] for item in art_one["sentences"]]
        titles.append(tt_one)
        art_ids.append(id_one)
        arts_sentences.append(sentences_one)
        art_event.append(art_one["event"])
        art_trigger.append(art_one["trigger"])
        art_who.append(art_one["who"])
        art_where.append(art_one["where"])
        art_when.append(art_one["when"])
        art_outcome.append(art_one["outcome"])

    pred_acs = art_cluster(args_, arts_sentences, titles, art_ids, args_.similar_max_min, art_event, art_trigger, art_who, art_where, art_when, art_outcome)
    print("Clustering done.")

    pred_acs = pred_acs[:-1]

    for ac_id, ac_values in enumerate(tqdm(pred_acs)):
        new_ac = list()
        count_sentences = 0.
        for art_id in ac_values:
            for sentences, id_, tt in zip(arts_sentences, art_ids, titles):
                if id_ == art_id:
                    new_ac.append((id_, sentences, tt, "Unknown"))
                    count_sentences += len(sentences)
        if args_.doc_ratio_type == 1 or count_sentences <= args_.rdoc_thr:
            new_doc = list()
            for doc_id, doc_one in enumerate(new_ac):
                publish_date = doc_one[3]
                new_doc.append(f"##Article {doc_id + 1} published on {publish_date}##")
                new_doc += doc_one[1]
                new_doc.append(f"##End of Article {doc_id + 1}##\n")
        else:
            new_doc, _ = rDoc(new_ac, doc_ratio=args_.doc_ratio, doc_ratio_type=args_.doc_ratio_type, hp1=args_.hp1, hp2=args_.hp2)
        summary = summarize_rdoc(new_doc)
        pred_summaries.append(summary)
    
    # gold summaries
    with open("data/mds_clusters_gpt_event.json", "r") as f:
        g_ac_list = json.load(f)
        gold_summaries = [ac["silver_truth"] for ac in g_ac_list]
    pipeline_rouge(pred_summaries, gold_summaries)

def eval(args_, do_cluster=True, do_summ=True):
    if not hit_summ_test and do_cluster:
        data_path = "data/mds_ac_test_final.json"
        print(args_.similar_max_min)
        with open(data_path, "r") as f:
            ac_test = json.load(f)
            arts = ac_test["articles"]
            arts_sentences, titles, art_ids, art_event, art_trigger, art_who, art_where, art_when, art_outcome = [], [], [], [], [], [], [], [], []
            for art_one in arts:
                arts_sentences.append([item["content"] for item in art_one["sentences"]])
                titles.append(art_one["title"])
                art_ids.append(art_one["ID"])
                art_event.append(art_one["event"])
                art_trigger.append(art_one["trigger"])
                art_who.append(art_one["who"])
                art_where.append(art_one["where"])
                art_when.append(art_one["when"])
                art_outcome.append(art_one["outcome"])
            goals = ac_test["clusters"]
        pred_acs = art_cluster(args_, arts_sentences, titles, art_ids, args_.similar_max_min, art_event, art_trigger, art_who, art_where, art_when, art_outcome)
        art_to_ac_id = dict()
        for ac_id, ac_values in enumerate(pred_acs):
            for art_id in ac_values:
                art_to_ac_id[art_id] = ac_id
        art_ids = sorted(art_to_ac_id.keys())
        pred_ac_labels = [art_to_ac_id[item] for item in art_ids]
        if args_.nmi_type == 2:
            count_other = goals.count(166)
            goals = goals[:-count_other]
            pred_ac_labels = pred_ac_labels[:-count_other]
        nmi_score = metrics.normalized_mutual_info_score(goals, pred_ac_labels)
        print(f"Test NMI score: {nmi_score}")

    if do_summ:
        pred_summaries, gold_summaries = list(), list()
        data_path = "data/mmds_test_hit_filtered.json" if hit_summ_test else "data/mds_clusters_gpt_event.json"
        with open(data_path, "r") as f:
            ac_list = json.load(f)
            ac_list_new = list()
            auto_data = list()
            for ac in tqdm(ac_list):
                docs = ac["documents"]
                new_ac = list()
                count_sentences = 0.
                for doc_one in docs:
                    sents = [item["content"].strip().replace("\n", " ") for item in doc_one["sentences"]]
                    article_id = doc_one["ID"]
                    org_title = doc_one["title"]
                    publish_date = doc_one["publish_date"]
                    new_ac.append((article_id, sents, org_title, publish_date))
                    count_sentences += len(sents)
                if args_.doc_ratio_type == 1 or count_sentences <= args_.rdoc_thr:
                    new_doc = list()
                    random.shuffle(new_ac)
                    for doc_id, doc_one in enumerate(new_ac):
                        publish_date = doc_one[3]
                        new_doc.append(f"##Article {doc_id + 1} published on {publish_date}##")
                        new_doc += doc_one[1]
                        new_doc.append(f"##End of Article {doc_id + 1}##\n")
                else:
                    new_doc, _ = rDoc(new_ac, doc_ratio=args_.doc_ratio, doc_ratio_type=args_.doc_ratio_type, hp1=args_.hp1, hp2=args_.hp2)

                summary = None
                while summary is None:
                    summary = summarize_rdoc(new_doc, gpt_use=args_.gpt_use)
                    if args_.doc_ratio_type == 3 and "<|endoftext|>" in summary:
                        summary = None  # post processing
                        continue
                gpt_summ = ac["silver_truth"]

                if "llm_local" in args_.summ_name:
                    key_name = args_.summ_name
                elif args_.doc_ratio_type == 1:
                    # Fine-tuned LLM: Baseline
                    key_name = "baseline"
                elif args_.doc_ratio_type == 2:
                    # Fine-tuned LLM: Highest Ent
                    key_name = f"predicted-{args_.doc_ratio}"
                else:
                    # Fine-tuned LLM: Highest Ent&Len
                    key_name = f"predicted-ent-len-{args_.hp1}"
                
                if args_.gpt_use:
                    gpt_name = "gpt35" if hit_summ_test else "gpt4"
                    key_name = key_name + f"-{gpt_name}"

                if args_.summ_name != "" and "llm_local" not in args_.summ_name:  # For raw models without FT
                    key_name = key_name + f"-{args_.summ_name}"

                if summary != "GPT: [OpenAI-Ban]":
                    ac[key_name] = summary
                    ac_list_new.append(ac.copy())
                    pred_summaries.append(summary)
                    gold_summaries.append(gpt_summ)
        with open(data_path, "w") as f:
            json.dump(ac_list_new, f, indent=1)
        r1_f, r2_f, rl_f = rouge_(pred_summaries, gold_summaries)
        print(f"Rouge-1: {r1_f}, -2: {r2_f}, -L: {rl_f}")

def eval_cluster(args_):
    nmi_score = 0.
    data_path = "data/mds_ac_dev_final.json"
    with open(data_path, "r") as f:
        ac_test = json.load(f)
        arts = ac_test["articles"]
        arts_sentences, titles, art_ids, art_event, art_trigger, art_who, art_where, art_when, art_outcome = [], [], [], [], [], [], [], [], []
        for art_one in arts:
            arts_sentences.append([item["content"] for item in art_one["sentences"]])
            titles.append(art_one["title"])
            art_ids.append(art_one["ID"])
            art_event.append(art_one["event"])
            art_trigger.append(art_one["trigger"])
            art_who.append(art_one["who"])
            art_where.append(art_one["where"])
            art_when.append(art_one["when"])
            art_outcome.append(art_one["outcome"])

        goals = ac_test["clusters"]
    pred_acs = art_cluster(args_, arts_sentences, titles, art_ids, args_.similar_max_min, art_event, art_trigger, art_who, art_where, art_when, art_outcome)
    art_to_ac_id = dict()
    for ac_id, ac_values in enumerate(pred_acs):
        for art_id in ac_values:
            art_to_ac_id[art_id] = ac_id
    art_ids = sorted(art_to_ac_id.keys())
    pred_ac_labels = [art_to_ac_id[item] for item in art_ids]
    if args_.nmi_type == 2:
        count_other = goals.count(166)
        goals = goals[:-count_other]
        pred_ac_labels = pred_ac_labels[:-count_other]
    nmi_score = metrics.normalized_mutual_info_score(goals, pred_ac_labels)
    return nmi_score

def eval_summ(args_):
    data_path = "data/mmds_valid_hit_filtered.json" if hit_summ_test else "data/mds_clusters_gpt_event_dev.json"
    with open(data_path, "r") as f:
        ac_list = json.load(f)
        ac_list_new = list()
        auto_data = list()
        for ac in tqdm(ac_list):
            docs = ac["documents"]
            new_ac = list()
            count_sentences = 0.
            for doc_one in docs:
                sents = [item["content"].strip().replace("\n", " ") for item in doc_one["sentences"]]
                article_id = doc_one["ID"]
                org_title = doc_one["title"]
                publish_date = doc_one["publish_date"]
                new_ac.append((article_id, sents, org_title, publish_date))
                count_sentences += len(sents)
            # Form new doc
            if args_.doc_ratio_type == 1 or count_sentences <= args_.rdoc_thr:
                new_doc = list()
                random.shuffle(new_ac)
                for doc_id, doc_one in enumerate(new_ac):
                    publish_date = doc_one[3]
                    new_doc.append(f"##Article {doc_id + 1} published on {publish_date}##")
                    new_doc += doc_one[1]
                    new_doc.append(f"##End of Article {doc_id + 1}##\n")
            else:
                new_doc, _ = rDoc(new_ac, doc_ratio=args_.doc_ratio, doc_ratio_type=args_.doc_ratio_type, hp1=args_.hp1, hp2=args_.hp2)
            summary = summarize_rdoc(new_doc, gpt_use=args_.gpt_use)
            
            if args_.summ_name == "llm_local":
                key_name = args_.summ_name
            elif args_.doc_ratio_type == 1:
                # Fine-tuned LLM: Baseline
                key_name = "baseline"
            elif args_.doc_ratio_type == 2:
                # Fine-tuned LLM: Highest Ent
                key_name = f"predicted-{args_.doc_ratio}"
            else:
                # Fine-tuned LLM: Highest Ent&Len
                key_name = f"predicted-ent-len-{args_.hp1}"
            
            if args_.gpt_use:
                gpt_name = "gpt35" if hit_summ_test else "gpt4"
                key_name = key_name + f"-{gpt_name}"

            if args_.summ_name != "" and args_.summ_name != "llm_local":  # For raw models without FT
                key_name = key_name + f"-{args_.summ_name}"

            ac[key_name] = summary
            ac_list_new.append(ac.copy())
    with open(data_path, "w") as f:
        json.dump(ac_list_new, f, indent=1)

def mmds_run(args_, article_lines):
    """ In stateless mode, will input some texts, each line is an article, we need to cluster the articles and then form summaries.
    """
    print("App started.")
    pred_summaries = list()

    # 1. clustering articles
    titles = list()
    art_ids = list()
    publishedDates = list()
    URLs = list()
    mes = list()
    arts_sentences = list()
    for id_one, art_one in enumerate(article_lines):
        if "ID" in art_one.keys():
            id_one = art_one["ID"]
        if "id" in art_one.keys():
            id_one = art_one["id"]

        if "event" not in art_one.keys() or "trigger" not in art_one.keys() or "who" not in art_one.keys() or "where" not in art_one.keys() or "when" not in art_one.keys() or "outcome" not in art_one.keys():
            art_one = get_me([art_one])[0]
        # high precision
        main_event = art_one["event"].strip() if "event" in art_one.keys() else ""
        trigger = art_one["trigger"].strip() if "trigger" in art_one.keys() else ""
        who = art_one["who"].strip() if "who" in art_one.keys() else ""
        where = art_one["where"].strip() if "where" in art_one.keys() else ""
        when = art_one["when"].strip() if "when" in art_one.keys() else ""
        outcome = art_one["outcome"].strip() if "outcome" in art_one.keys() else ""

        art_tt = art_one["title"].strip() if "title" in art_one.keys() else ""
        art_url = art_one["URL"].strip() if "URL" in art_one.keys() else "Unknown"
        art_pub_date = art_one["publishedDate"] if "publishedDate" in art_one.keys() else "Unknown"
        sentences_one = art_one["sentences"] if "sentences" in art_one.keys() else ""
        if len(sentences_one) == 0:
            art_content = art_one["content"].strip()
            if len(art_content) == 0:
                continue
            sentences_one = split_sentences(art_content)
        llm_me = f'The event of "{main_event}" happened in {where} on {when}, it is related to {who}.'
        mes.append(llm_me)
        titles.append(art_tt)
        art_ids.append(id_one)
        publishedDates.append(art_pub_date)
        URLs.append(art_url)
        arts_sentences.append(sentences_one)

    pred_acs = art_cluster(args_, arts_sentences, titles, mes, art_ids, args_.similar_max_min)
    ac_list = list()
    for ac_values in pred_acs:
        # form ac sentences
        ac_sentences = list()
        for art_id in ac_values:
            for sentences, id_, tt, pub, url in zip(arts_sentences, art_ids, titles, publishedDates, URLs):
                if id_ == art_id:
                    ac_sentences.append((id_, sentences, tt, pub, url))
        ac_list.append(ac_sentences)
    print("App cluster done.")
    return ac_list

def merge_to_main(merge_to_path, add_file, key_name=None):
    with open(merge_to_path, "r") as f1:
        main_list = json.load(f1)
    with open(add_file, "r") as f2:
        addt_list = json.load(f2)

    merge_list = list()
    for ac_1, ac_2 in zip(main_list, addt_list):
        if key_name is None:
            for add_key in ac_2.keys():
                if add_key in ac_1.keys():
                    continue
                else:
                    ac_1[add_key] = ac_2[add_key]
        else:
            for add_key in ac_2.keys():
                if add_key in key_name:
                    ac_1[add_key] = ac_2[add_key]
        merge_list.append(ac_1.copy())
    with open(merge_to_path, "w") as fw:
        json.dump(merge_list, fw, indent=1)

def mmds_main(request_arts=None, evaluate_flag=3, doc_ratio_type=1, dev_use=False, gpt_use=False, summ_name="", use_hit=True, cluster_type=1, nmi_type=1, rdoc_thr=90, load_lora=True, cversion=1):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--evaluate", default=False, type=bool, help="Evaluate the two-stage system or not.")
    arg_parser.add_argument("--eval_pipeline", default=False, type=bool, help="Evaluate the pipeline system or not.")
    arg_parser.add_argument("--clustering_only", default=clustering_only, type=bool, help="If True, will not summarize for each comment cluster.")

    arg_parser.add_argument("--article_cluster", default=None)
    arg_parser.add_argument("--cversion", default=cversion, type=int)
    arg_parser.add_argument("--cword_rep", default="sentence-transformer")  # glove, transformer, sentence-transformer
    arg_parser.add_argument("--cfeature_space", default=1, type=int)
    arg_parser.add_argument("--cfeature_label", default=3, type=int)
    arg_parser.add_argument("--cmt_id_feature", default=True, type=bool, help="Indicate whether two semantic spans belong to one comment or not.")
    arg_parser.add_argument("--ckw_feature", default=True, type=bool, help="Whether use the representation of keywords for clustering or not.")
    arg_parser.add_argument("--comment_span", default=use_comment_span, type=bool, help="Combine continuous sentences inside a comment as semantic spans.")
    arg_parser.add_argument("--dynamic_similar_thr", default=True, type=bool, help="Use dynamic similarity or not.")
    arg_parser.add_argument("--similar_max_min", default=None)  # from 0.5--50 to 0.6--400 comments
    arg_parser.add_argument("--w2v_size", default=128)
    arg_parser.add_argument("--cmt_truncation", default=comment_truncation, type=bool)
    arg_parser.add_argument("--use_ac_tt", default=use_ac_tt, type=bool)
    arg_parser.add_argument("--use_ac_t3", default=use_ac_t3, type=bool)
    arg_parser.add_argument("--use_ac_1st", default=use_ac_1st, type=bool)
    arg_parser.add_argument("--use_ac_me", default=use_ac_me, type=bool)
    arg_parser.add_argument("--use_llm_me", default=use_llm_me, type=bool)
    arg_parser.add_argument("--use_event", default=use_event, type=bool)
    arg_parser.add_argument("--use_trigger", default=use_trigger, type=bool)
    arg_parser.add_argument("--use_who", default=use_who, type=bool)
    arg_parser.add_argument("--use_where", default=use_where, type=bool)
    arg_parser.add_argument("--use_when", default=use_when, type=bool)
    arg_parser.add_argument("--use_outcome", default=use_outcome, type=bool)
    # evaluation attributions
    arg_parser.add_argument("--use_db", default=use_db)
    arg_parser.add_argument("--bg_icl", default=True, type=bool)
    arg_parser.add_argument("--llm_bg", default=llm_based_background, type=bool)
    arg_parser.add_argument("--doc_ratio", default=0.6, type=float)
    arg_parser.add_argument("--doc_ratio_type", default=doc_ratio_type, type=int)  # 0 for static, 1 for ent + len
    arg_parser.add_argument("--cluster_type", default=cluster_type, type=int)
    arg_parser.add_argument("--nmi_type", default=nmi_type, type=int)  # 1 for full, 2 for pure
    arg_parser.add_argument("--hp1", default=0.5, type=float)
    arg_parser.add_argument("--hp2", default=0.5, type=float)
    arg_parser.add_argument("--rdoc_thr", default=rdoc_thr, type=int)
    arg_parser.add_argument("--dev_use", default=dev_use, type=bool)
    arg_parser.add_argument("--gpt_use", default=gpt_use, type=bool)
    arg_parser.add_argument("--summ_name", default=summ_name)
    arg_parser.set_defaults(use_gpu=True)
    args_ = arg_parser.parse_args()
    global hit_summ_test
    hit_summ_test = use_hit

    # Hyper-parameters for clustering
    if clustering_thr is not None:
        args_.similar_max_min = clustering_thr
    elif cluster_type == 1:
        args_.similar_max_min = [0.52, 0.95, 0.52, 12]
    else:
        args_.similar_max_min = [0.5, 0.95, 0.54, 12]
    
    if use_hit:
        args_.doc_ratio = 0.4
        args_.hp1 = 0.3
    else:
        args_.doc_ratio = 0.5
        args_.hp1 = 0.8
    
    if not gpt_use:
        load_tqa = (summ_name!="" and "llm_local" in summ_name)
        load_model(load_tqa, load_lora)

    if evaluate_flag == 1:
        if dev_use:
            if not hit_summ_test:
                if cluster_type == 1:
                    # Step 1. basic clustering threshold selection, employing the basic ME references
                    highest_flag = -100000
                    parameter_info = None
                    for ratio_ in range(0, 6):
                        begin_thr = 0.5 + ratio_ * 0.02
                        end_thr = begin_thr
                        args_.similar_max_min = [begin_thr, 0.95, end_thr, 12]
                        nmi_value = eval_cluster(args_)
                        print([begin_thr, 0.95, end_thr, 12], nmi_value)
                        if nmi_value > highest_flag:
                            parameter_info = [begin_thr, 0.95, end_thr, 12]
                            highest_flag = nmi_value
                    print("Cluster hyper-parameters: ", parameter_info, highest_flag)
                    args_.similar_max_min = parameter_info
                elif cluster_type == 2:
                    # Step 2. dy clustering threshold selection
                    highest_flag = -100000
                    parameter_info = None
                    for ratio_ in range(0, 6):
                        begin_thr = 0.5 + ratio_ * 0.02
                        for ratio_2 in range(1, 11):
                            end_thr = min(begin_thr + ratio_2 * 0.02, 0.95)
                            args_.similar_max_min = [begin_thr, 0.95, end_thr, 12]
                            nmi_value = eval_cluster(args_)
                            if nmi_value > highest_flag:
                                parameter_info = [begin_thr, 0.95, end_thr, 12]
                                highest_flag = nmi_value
                            print([begin_thr, 0.95, end_thr, 12], nmi_value)
                    print("Cluster hyper-parameters: ", parameter_info, highest_flag)
                    args_.similar_max_min = parameter_info
                else:
                    # Step 3. clustering features selection
                    nmi_value = eval_cluster(args_)
                    print("Dev NMI score: ", nmi_value)
                eval(args_, do_summ=False)

            # mmds
            if args_.doc_ratio_type == 2:
                # 1. Certain rate for each ac
                for ratio_ in range(1, 11):
                    args_.doc_ratio = round(ratio_ * 0.1, 2)
                    eval_summ(args_)
            else:
                # 2. Highest entropy + length
                for ratio_ in range(1, 11):
                    args_.hp1 = round(ratio_ * 0.1, 2)
                    args_.hp2 = 1.0 - args_.hp1
                    eval_summ(args_)
        else:
            eval(args_, do_cluster=True)
    elif evaluate_flag == 2:
        pipe_eval(args_)
    else:
        ac_list = mmds_run(args_, request_arts)
        return ac_list

def get_max_results(file_path="test"):
    print(file_path)
    with open(f"data/mmds_{file_path}_hit.json", "r") as f:
        instances = json.load(f)
        instances = instances[::-1]
        last_tt = None
        filtered_instances = []
        for ins in instances:
            tmp_tt = ins["documents"][0]["title"]
            if tmp_tt == last_tt:
                continue
            last_tt = tmp_tt
            filtered_instances.append(ins.copy())
    with open(f"data/mmds_{file_path}_hit_filtered.json", "w") as w:
        json.dump(filtered_instances, w, indent=1)


if __name__ == '__main__':
    """ evaluate_flag:  1 for eval, 2 for pipeline eval, 3 for test
        doc_ratio_type: 1 for baseline, 2 for static rate, 3 for highest rate + length
    """
    # =============================================================================
    # HIT dataset
    # get_max_results(file_path="valid")
    # get_max_results(file_path="test")

    # Dev
    # mmds_main(evaluate_flag=1, doc_ratio_type=2, dev_use=True, use_hit=True)  # ent
    # mmds_main(evaluate_flag=1, doc_ratio_type=3, dev_use=True, use_hit=True)  # ent_len
    # mmds_main(evaluate_flag=1, doc_ratio_type=2, dev_use=True, use_hit=True, load_lora=False, summ_name="raw_qwen")  # ent-raw
    # mmds_main(evaluate_flag=1, doc_ratio_type=3, dev_use=True, use_hit=True, load_lora=False, summ_name="raw_qwen")  # ent_len-raw
    # mmds_main(evaluate_flag=1, doc_ratio_type=2, dev_use=True, gpt_use=True, use_hit=True)   # GPT35-ent
    # mmds_main(evaluate_flag=1, doc_ratio_type=3, dev_use=True, gpt_use=True, use_hit=True)   # GPT35-ent_len
    
    # mmds_main(evaluate_flag=1, doc_ratio_type=3, dev_use=True, use_hit=True, summ_name="llm_local")  # llm_local need to select hp also, give up

    # Test
    # mmds_main(evaluate_flag=1, doc_ratio_type=1, dev_use=False, use_hit=True)   # baseline
    # mmds_main(evaluate_flag=1, doc_ratio_type=2, dev_use=False, use_hit=True)  # ent
    # mmds_main(evaluate_flag=1, doc_ratio_type=3, dev_use=False, use_hit=True)  # ent_len
    # mmds_main(evaluate_flag=1, doc_ratio_type=3, dev_use=False, use_hit=True, summ_name="llm_local")  # ent_len localized
    # mmds_main(evaluate_flag=1, doc_ratio_type=1, dev_use=False, use_hit=True, load_lora=False, summ_name="raw_qwen")   # baseline + raw qwen
    # mmds_main(evaluate_flag=1, doc_ratio_type=2, dev_use=False, use_hit=True, load_lora=False, summ_name="raw_qwen")   # ent + raw qwen
    # mmds_main(evaluate_flag=1, doc_ratio_type=3, dev_use=False, use_hit=True, load_lora=False, summ_name="raw_qwen")   # ent_len + raw qwen
    
    # mmds_main(evaluate_flag=1, doc_ratio_type=1, dev_use=False, gpt_use=True, use_hit=True)   # GPT35
    # mmds_main(evaluate_flag=1, doc_ratio_type=2, dev_use=False, gpt_use=True, use_hit=True)   # GPT35-ent
    # mmds_main(evaluate_flag=1, doc_ratio_type=3, dev_use=False, gpt_use=True, use_hit=True)   # GPT35-ent_len
    # mmds_main(evaluate_flag=1, doc_ratio_type=1, dev_use=False, gpt_use=True, use_hit=True)   # GPT4 pls manually update gpt35 to 4
    
    # =============================================================================
    # Our dataset
    # Dev Clustering
    # Step 1. Baseline: Raw fast clustering + title & first three sentences as reference
    # mmds_main(evaluate_flag=1, doc_ratio_type=1, cluster_type=1, dev_use=True, use_hit=False, nmi_type=1)  # 0.54 best for baseline, test 
    # mmds_main(evaluate_flag=1, doc_ratio_type=1, cluster_type=1, dev_use=True, use_hit=False, nmi_type=2)  # 0.54 best for baseline
    
    # Step 2. Baseline + Dy
    # mmds_main(evaluate_flag=1, doc_ratio_type=1, cluster_type=2, dev_use=True, use_hit=False, nmi_type=1)
    # mmds_main(evaluate_flag=1, doc_ratio_type=1, cluster_type=2, dev_use=True, use_hit=False, nmi_type=2)

    # Step 3. Baseline + Dy + Reference feature Selection
    # mmds_main(evaluate_flag=1, doc_ratio_type=1, cluster_type=2, dev_use=True, use_hit=False, nmi_type=1)
    # mmds_main(evaluate_flag=1, doc_ratio_type=1, cluster_type=2, dev_use=True, use_hit=False, nmi_type=2)
    
    # Step 4. DEV Summarization
    # mmds_main(evaluate_flag=1, doc_ratio_type=1, dev_use=True, use_hit=False, rdoc_thr=70)  # baseline for picture drawing
    # mmds_main(evaluate_flag=1, doc_ratio_type=2, dev_use=True, use_hit=False, rdoc_thr=70)  # ent, rdoc_thr=70
    # mmds_main(evaluate_flag=1, doc_ratio_type=3, dev_use=True, use_hit=False, rdoc_thr=70)  # ent_len
    # mmds_main(evaluate_flag=1, doc_ratio_type=3, dev_use=True, use_hit=False, rdoc_thr=70, summ_name="llm_local")  # ent_len localized (not run)
    # mmds_main(evaluate_flag=1, doc_ratio_type=3, dev_use=True, use_hit=False, rdoc_thr=70, summ_name="llm_local_sg")  # ent_len localized_sg
    
    # Test | Baseline and GPT no change, no need to re-run
    # mmds_main(evaluate_flag=1, doc_ratio_type=1, dev_use=False, use_hit=False, rdoc_thr=70)  # baseline
    # mmds_main(evaluate_flag=1, doc_ratio_type=2, dev_use=False, use_hit=False, rdoc_thr=70)  # ent
    # mmds_main(evaluate_flag=1, doc_ratio_type=3, dev_use=False, use_hit=False, rdoc_thr=70)  # ent_len
    # mmds_main(evaluate_flag=1, doc_ratio_type=3, dev_use=False, use_hit=False, rdoc_thr=70, summ_name="llm_local")  # ent_len localized
    # mmds_main(evaluate_flag=1, doc_ratio_type=3, dev_use=False, use_hit=False, rdoc_thr=70, summ_name="llm_local_sg")  # ent_len localized_sg
    mmds_main(evaluate_flag=1, doc_ratio_type=3, dev_use=False, use_hit=False, rdoc_thr=70, summ_name="ctx")  # ent_len ctx limited
    mmds_main(evaluate_flag=1, doc_ratio_type=3, dev_use=False, use_hit=False, rdoc_thr=70, summ_name="llm_local_ctx")  # ent_len localized_sg ctx limited
    # mmds_main(evaluate_flag=1, doc_ratio_type=1, dev_use=False, use_hit=False, rdoc_thr=70, load_lora=False, summ_name="raw_seallm")  # baseline + raw seallm
    # mmds_main(evaluate_flag=1, doc_ratio_type=2, dev_use=False, use_hit=False, rdoc_thr=70, load_lora=False, summ_name="raw_seallm")  # baseline + raw seallm
    # mmds_main(evaluate_flag=1, doc_ratio_type=3, dev_use=False, use_hit=False, rdoc_thr=70, load_lora=False, summ_name="raw_seallm")  # baseline + raw seallm
    
    # Test Pipeline
    # V1: FC w/ TT-T3 + Basic MMDS
    # mmds_main(evaluate_flag=2, doc_ratio_type=1, cluster_type=1, dev_use=False, use_hit=False, rdoc_thr=70)
    # V2: DyClu w/ Best-Feats + Basic MMDS
    # mmds_main(evaluate_flag=2, doc_ratio_type=1, cluster_type=2, dev_use=False, use_hit=False, rdoc_thr=70)
    # V3: DyClu w/ Best-Feats + Enhanced MMDS  
    # mmds_main(evaluate_flag=2, doc_ratio_type=3, cluster_type=2, dev_use=False, use_hit=False, rdoc_thr=70)
    # V4: DyClu w/ Best-Feats + Enhanced MMDS + Localization
    # mmds_main(evaluate_flag=2, doc_ratio_type=3, cluster_type=2, dev_use=False, use_hit=False, rdoc_thr=70, summ_name="llm_local")

    # =============================================================================
    # Our test
    # mmds_main(evaluate_flag=1, doc_ratio_type=1, cluster_type=2, dev_use=False, use_hit=False, nmi_type=1)
    # mmds_main(evaluate_flag=1, doc_ratio_type=1, cluster_type=2, dev_use=False, use_hit=False, nmi_type=2)

    # goals = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20]
    # pred_ac_labels = [2, 2, 2, 2, 2, 6, 6, 6, 6, 6, 10, 10, 10, 10, 10, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 13, 13, 13, 13, 13, 11, 11, 11, 11, 11, 14, 14, 14, 14, 20, 16, 20, 16, 16, 16, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 12, 12, 12, 12, 12, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 0, 0, 20, 0, 0, 15, 20, 15, 15, 15, 7, 7, 7, 7, 7, 17, 20, 17, 17, 17, 18, 19, 18, 19, 18]
    # pred_ac_labels2 = [2, 2, 2, 2, 2, 6, 6, 6, 6, 6, 10, 10, 10, 10, 10, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 13, 13, 13, 13, 13, 11, 11, 11, 11, 11, 14, 14, 14, 14, 20, 16, 21, 16, 16, 16, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 12, 12, 12, 12, 12, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 0, 0, 22, 0, 0, 15, 23, 15, 15, 15, 7, 7, 7, 7, 7, 17, 24, 17, 17, 17, 18, 19, 18, 19, 18]
    # nmi_score = metrics.normalized_mutual_info_score(goals, pred_ac_labels)
    # print(nmi_score)
    # nmi_score = metrics.normalized_mutual_info_score(goals, pred_ac_labels2)
    # print(nmi_score)
    
    # BGE
    # [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20]
    # [5, 5, 5, 5, 5, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 13, 13, 13, 13, 13, 10, 10, 10, 10, 10, 14, 14, 14, 14, 14, 17, 17, 17, 17, 17, 12, 12, 12, 12, 12, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 6, 6, 6, 6, 6, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 11, 11, 11, 11, 11, 1, 1, 1, 1, 18, 15, 15, 15, 15, 15, 1, 1, 18, 1, 1]
    # 96.8
    # goals = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20]
    # pred_ac_labels = [5, 5, 5, 5, 5, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 13, 13, 13, 13, 13, 10, 10, 10, 10, 10, 14, 14, 14, 14, 14, 17, 17, 17, 17, 17, 12, 12, 12, 12, 12, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 6, 6, 6, 6, 6, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 11, 11, 11, 11, 11, 1, 1, 1, 1, 18, 15, 15, 15, 15, 15, 1, 1, 19, 1, 1]
    # nmi_score = metrics.normalized_mutual_info_score(goals, pred_ac_labels)
    # 97.05