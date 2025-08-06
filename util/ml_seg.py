import warnings
warnings.filterwarnings('ignore') 

from nltk import pos_tag
from nltk.tokenize import word_tokenize
import jieba.posseg as pseg
import langid
import malaya

bmi_model = None

def language_detection(sentence):
    lang_id = langid.classify(sentence)[0]
    lang_dict = {"ms": "BM", "zh": "CN", "id": "BI", "en": "EN"}
    if lang_id in lang_dict.keys():
        return lang_dict[lang_id]
    else:
        return "EN"

def pos_tag_sentences(comments):
    global bmi_model
    if bmi_model is None:
        bmi_model = malaya.pos.transformer(model='xlnet')

    tagged_sentences = []
    for sentence in comments:
        sent_lang = language_detection(sentence)

        # English POS tagging
        if sent_lang == "EN":
            english_tokens = word_tokenize(sentence)
            tags = pos_tag(english_tokens)

        # Chinese POS tagging
        if sent_lang == "CN":
            chinese_tags = pseg.cut(sentence)
            tags = [(word, tag) for word, tag in chinese_tags]

        if sent_lang in ["BM", "BI"]:
            tags = bmi_model.predict(sentence)
        tagged_sentences.append(tags)
    return tagged_sentences
