from peft import PeftModel
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, GenerationConfig
from tqdm import tqdm
import torch
import json
from rouge_score import rouge_scorer
from azure import gpt_ann, gpt_response

base_model_path = "ml_models/seallm_7b/7B_V2"
tuned_lora = "ml_models/checkpoints/mmds_me/checkpoint-2000/"
llm_tokenizer = AutoTokenizer.from_pretrained(base_model_path, padding_side='left')
llm_tokenizer.add_special_tokens = False

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model_ = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16, quantization_config=bnb_config, device_map=f"cuda")
model_.config.use_cache = False
model_lora = PeftModel.from_pretrained(model_, tuned_lora, torch_dtype=torch.bfloat16)
model_lora = model_lora.to(f"cuda")
model_lora.eval()
generation_config = GenerationConfig(do_sample=True, temperature=0.7, top_p=1.0, top_k=5, num_beams=2, eos_token_id=llm_tokenizer.eos_token_id, pad_token_id=llm_tokenizer.eos_token_id, pad_token=llm_tokenizer.pad_token)

def seallm_style(ask):
    return f"<|im_start|>system \nYou are a helpful assistant.</s><|im_start|>user \nHello world</s><|im_start|>assistant \nHi there, how can I help you today?</s><|im_start|>user \n{ask}</s>"

def get_me(articles):
    new_articles = list()
    for article in tqdm(articles):
        article_content = article["content"].replace("\n", " ")
        me_prompt = f"**Article:**\n{article_content}\n\n**Instructions:**\nPlease review the article, determine the main event of the article, and extract the event trigger, arguments (Who/What, Where, When), and outcome in English. Return the main event information in the format of 'Event: ... | Trigger: ... | Who/What: ... | Where: ... | When: ... | Outcome: ...', making the summary more accurately reflects the content and details of the article.\n\n**Main event information:**"
        
        me_prompt = seallm_style(me_prompt)

        me_inputs = llm_tokenizer(me_prompt, padding=True, return_tensors="pt", max_length=2048, truncation=True)
        me_ids = me_inputs["input_ids"].to("cuda")
        generation_output = model_lora.generate(input_ids=me_ids, generation_config=generation_config, return_dict_in_generate=True, output_scores=True, max_new_tokens=1000)
        s = generation_output.sequences[0]
        response_ = llm_tokenizer.decode(s).strip()
        response = response_.split("</s>")[-2].replace("\n", " ").strip()
        response = response.replace("event:", "Event:")
        response = response.replace("trigger:", "Trigger:")
        response = response.replace("who/what:", "Who/What:")
        response = response.replace("where:", "Where:")
        response = response.replace("when:", "When:")
        response = response.replace("outcome:", "Outcome:")

        if "Outcome:" in response:
            outcome = response.split("Outcome:")[-1].strip()
            response = "Outcome:".join(response.split("Outcome:")[:-1])
        else:
            outcome = "NA"

        if "When:" in response:
            when = response.split("When:")[1].replace("|", "").strip()
            response = "When:".join(response.split("When:")[:-1])
        else:
            when = "NA"

        if "Where:" in response:
            where = response.split("Where:")[1].replace("|", "").strip()
            response = "Where:".join(response.split("Where:")[:-1])
        else:
            where = "NA"

        if "Who/What:" in response:
            who = response.split("Who/What:")[1].replace("|", "").strip()
            response = "Who/What:".join(response.split("Who/What:")[:-1])
        else:
            who = "NA"

        if "Trigger:" in response:
            trigger = response.split("Trigger:")[1].replace("|", "").strip()
            response = "Trigger:".join(response.split("Trigger:")[:-1])
        else:
            trigger = "NA"

        if "Event:" in response:
            event = response.split("Event:")[1].replace("|", "").strip()
            response = "Event:".join(response.split("Event:")[:-1])
        else:
            event = "NA"
        article["event"] = event
        article["trigger"] = trigger
        article["who"] = who
        article["where"] = where
        article["when"] = when
        article["outcome"] = outcome
        new_articles.append(article.copy())
    return new_articles

def prepare_test_me(data_path):
    with open(data_path, "r") as f:
        test_acs = json.load(f)
    articles = test_acs["articles"]
    new_articles = list()
    for article in tqdm(articles):
        sentences = article["sentences"]
        article_content = " ".join([item["content"] for item in sentences])
        me_prompt = f"**Article:**\n{article_content}\n\n**Instructions:**\nPlease review the article, determine the main event of the article, and extract the event trigger, arguments (Who/What, Where, When), and outcome in English. Return the main event information in the format of 'Event: ... | Trigger: ... | Who/What: ... | Where: ... | When: ... | Outcome: ...', making the summary more accurately reflects the content and details of the article.\n\n**Main event information:**"
        me_prompt = seallm_style(me_prompt)
        me_inputs = llm_tokenizer(me_prompt, padding=True, return_tensors="pt", max_length=2048, truncation=True)
        me_ids = me_inputs["input_ids"].to("cuda")
        generation_output = model_lora.generate(input_ids=me_ids, generation_config=generation_config, return_dict_in_generate=True, output_scores=True, max_new_tokens=1000)
        s = generation_output.sequences[0]
        response_ = llm_tokenizer.decode(s).strip()
        response = response_.split("</s>")[-2].replace("\n", " ").strip()
        response = response.replace("event:", "Event:").replace("trigger:", "Trigger:").replace("who/what:", "Who/What:").replace("where:", "Where:").replace("when:", "When:").replace("outcome:", "Outcome:")
        if "Outcome:" in response:
            outcome = response.split("Outcome:")[-1].strip()
            response = "Outcome:".join(response.split("Outcome:")[:-1])
        else:
            outcome = "NA"

        if "When:" in response:
            when = response.split("When:")[1].replace("|", "").strip()
            response = "When:".join(response.split("When:")[:-1])
        else:
            when = "NA"

        if "Where:" in response:
            where = response.split("Where:")[1].replace("|", "").strip()
            response = "Where:".join(response.split("Where:")[:-1])
        else:
            where = "NA"

        if "Who/What:" in response:
            who = response.split("Who/What:")[1].replace("|", "").strip()
            response = "Who/What:".join(response.split("Who/What:")[:-1])
        else:
            who = "NA"

        if "Trigger:" in response:
            trigger = response.split("Trigger:")[1].replace("|", "").strip()
            response = "Trigger:".join(response.split("Trigger:")[:-1])
        else:
            trigger = "NA"

        if "Event:" in response:
            event = response.split("Event:")[1].replace("|", "").strip()
            response = "Event:".join(response.split("Event:")[:-1])
        else:
            event = "NA"
        article["event"] = event
        article["trigger"] = trigger
        article["who"] = who
        article["where"] = where
        article["when"] = when
        article["outcome"] = outcome
        new_articles.append(article.copy())
    test_acs["articles"] = new_articles[:]
    with open(data_path, "w") as w:
        json.dump(test_acs, w, indent=1)


def prepare_gpt_me(data_path):
    with open(data_path, "r") as f:
        test_acs = json.load(f)
    articles = test_acs["articles"]
    new_articles = list()
    for article in tqdm(articles):
        sentences = article["sentences"]
        article_content = " ".join([item["content"] for item in sentences])
        me_prompt = f"**Article:**\n{article_content}\n\n**Instructions:**\nPlease review the article, determine the main event of the article, and extract the event trigger, arguments (Who/What, Where, When), and outcome in English. Return the main event information in the format of 'Event: ... | Trigger: ... | Who/What: ... | Where: ... | When: ... | Outcome: ...', making the summary more accurately reflects the content and details of the article.\n\n**Main event information:**"
        response = gpt_response(me_prompt, gpt_type="gpt4")
        response = response.replace("event:", "Event:").replace("trigger:", "Trigger:").replace("who/what:", "Who/What:").replace("where:", "Where:").replace("when:", "When:").replace("outcome:", "Outcome:")
        if "Outcome:" in response:
            outcome = response.split("Outcome:")[-1].strip()
            response = "Outcome:".join(response.split("Outcome:")[:-1])
        else:
            outcome = "NA"

        if "When:" in response:
            when = response.split("When:")[1].replace("|", "").strip()
            response = "When:".join(response.split("When:")[:-1])
        else:
            when = "NA"

        if "Where:" in response:
            where = response.split("Where:")[1].replace("|", "").strip()
            response = "Where:".join(response.split("Where:")[:-1])
        else:
            where = "NA"

        if "Who/What:" in response:
            who = response.split("Who/What:")[1].replace("|", "").strip()
            response = "Who/What:".join(response.split("Who/What:")[:-1])
        else:
            who = "NA"

        if "Trigger:" in response:
            trigger = response.split("Trigger:")[1].replace("|", "").strip()
            response = "Trigger:".join(response.split("Trigger:")[:-1])
        else:
            trigger = "NA"

        if "Event:" in response:
            event = response.split("Event:")[1].replace("|", "").strip()
            response = "Event:".join(response.split("Event:")[:-1])
        else:
            event = "NA"
        article["event_gpt"] = event
        new_articles.append(article.copy())
    test_acs["articles"] = new_articles[:]
    with open(data_path, "w") as w:
        json.dump(test_acs, w, indent=1)

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

def rouge_eval(data_path):
    gold_mes, gpt_mes, our_mes = [], [], []
    with open(data_path, "r") as f:
        articles_obj = json.load(f)
        articles = articles_obj["articles"]
        for article_one in articles:
            sentences = article_one["sentences"]
            main_event = ""
            for sent_one in sentences:
                if sent_one["event_annotation"] == "main":
                    main_event = sent_one["content"]
                    break
            # rouge eval
            gpt_me = article_one["event_gpt"]
            our_me = article_one["event"]
            gpt_mes.append(gpt_me)
            our_mes.append(our_me)
            gold_mes.append(main_event)
    # test
    rouge_1 = rouge_(gpt_mes, gold_mes)
    rouge_2 = rouge_(our_mes, gold_mes)
    print(rouge_1)
    print(rouge_2)

if __name__ == '__main__':
    prepare_test_me(data_path="data/mds_ac_dev_final.json")
    prepare_test_me(data_path="data/mds_ac_test_final.json")
    
    # compare with GPT4
    # prepare_gpt_me(data_path="data/mds_ac_test_final.json")
    # rouge_eval(data_path="data/mds_ac_test_final.json")