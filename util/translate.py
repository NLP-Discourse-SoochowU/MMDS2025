import requests
import json
import time
import os
import random
import googletrans
from googletrans import Translator
import langid
translator = Translator()

def translate_(input_texts, tg_type):
    if tg_type == "bi":
        # url = "http://10.2.56.46:8001/predictions/en-id"
        url = "http://10.2.56.46:8000/predictions/id-en"
    elif tg_type == "zh":
        url = "http://10.2.56.210:8090/predictions/en-zh"
    else:
        url = None
    data_dict = {
        "data": input_texts
    }
    json_data = json.dumps(data_dict)
    headers = {
        "Content-Type": "application/json"
    }
    try:
        # Make the POST request
        response = requests.post(url, data=json_data, headers=headers)
        # Check if the request was successful (status code 2xx)
        if response.status_code // 100 == 2:
            translated = json.loads(response.text)
            return translated["translation"]
        else:
            print(f"Error posting data. Status code: {response.status_code}")
            print(response.text)
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

# results = translate_(["Dortmund VS Freiburg: Menang 5-1, Liga Jerman", "Desain Futuristik Kapal Perusak AS Disebut Mirip Kapal Perang China"], "bi")
# print(results)

def language_detection(sentence, hit=False):
    lang_id = langid.classify(sentence)[0]
    if hit:
        return lang_id
    lang_dict = {"ms": "BM", "zh": "CN", "id": "BI", "en": "EN"}
    if lang_id in lang_dict.keys():
        return lang_dict[lang_id]
    else:
        return "EN"

def google_trans(text_to_tran):
    sent_lang = language_detection(text_to_tran)
    if sent_lang == "CN":
        tg_cmt = translator.translate(text_to_tran, src="zh-cn", dest="en")
        tg_cmt = tg_cmt.text
    elif sent_lang == "BI":
        tg_cmt = translator.translate(text_to_tran, src="id", dest="en")
        tg_cmt = tg_cmt.text
    elif sent_lang == "BM":
        tg_cmt = translator.translate(text_to_tran, src="ms", dest="en")
        tg_cmt = tg_cmt.text
    else:
        tg_cmt = text_to_tran
    return tg_cmt

def google_trans_hit(text_to_tran):
    source_language = language_detection(text_to_tran, True)
    try:
        tg_cmt = translator.translate(text_to_tran, src=source_language, dest="en")
        tg_cmt = tg_cmt.text
        return tg_cmt
    except Exception as e:  # Handle potential translation errors
        print(f"Translation error: {text_to_tran}")
        return text_to_tran  # Return original text on error