import torch
import random
import spacy
import re 
from langdetect import detect  # Install with: pip install langdetect

# Load SpaCy language models
nlp_en = spacy.load("en_core_web_sm")
nlp_zh = spacy.load("zh_core_web_sm")

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def ml_encoding(sentences, tokenizer, model, device="cuda"):
    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return sentence_embeddings


bos_token = '<s>'
eos_token = '</s>'

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

sys_prompt = """You are a multilingual, helpful, respectful and honest assistant. \
Please always answer as helpfully as possible, while being safe. Your \
answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure \
that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not \
correct. If you don't know the answer to a question, please don't share false information.

As a multilingual assistant, you must respond and follow instructions in the native language of the user by default, unless told otherwise. \
Your response should adapt to the norms and customs of the respective language and culture.
"""

def chat_multiturn_seq_format(message, history):
    """
    ```
        <bos>[INST] B_SYS SytemPrompt E_SYS Prompt [/INST] Answer <eos>
        <bos>[INST] Prompt [/INST] Answer <eos>
        <bos>[INST] Prompt [/INST]
    ```
    As the format auto-add <bos>, please turn off add_special_tokens with `tokenizer.add_special_tokens = False`
    Inputs:
      message: the current prompt
      history: list of list indicating previous conversation. [[message1, response1], [message2, response2]]
    Outputs:
      full_prompt: the prompt that should go into the chat model

    e.g:
      full_prompt = chat_multiturn_seq_format("Hello world")
      output = model.generate(tokenizer.encode(full_prompt, add_special_tokens=False), ...)
    """
    text = ''
    for i, (prompt, res) in enumerate(history):
        if i == 0:
            text += f"{bos_token}{B_INST} {B_SYS} {sys_prompt} {E_SYS} {prompt} {E_INST}"
        else:
            text += f"{bos_token}{B_INST} {prompt}{end_instr}"
        if res is not None:
            text += f" {res} {eos_token} "
    if len(history) == 0 or text.strip() == '':
        text = f"{bos_token}{B_INST} {B_SYS} {sys_prompt} {E_SYS} {message} {E_INST}"
    else:
        text += f"{bos_token}{B_INST} {message} {E_INST}"
    return text

def chat_multiturn_seq_format_v1(message, history):
    """
    <s>[INST] <<SYS>>\n You are a multilingual, helpful, respectful and honest assistant. If you don't know the answer to a question, please don't share false information. As a multilingual assistant, your response should adapt to the norms and customs of the respective language and culture. \n<</SYS>>\n\n xxxxxxxxxxxx [/INST]
    """
    sys_prompt = """You are a multilingual, helpful, respectful and honest assistant. If you don't know the answer to a question, please don't share false information. As a multilingual assistant, your response should adapt to the norms and customs of the respective language and culture."""
    bos_token = '<s>'
    eos_token = '</s>'

    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    
    text = ''
    for i, (prompt, res) in enumerate(history):
        if i == 0:
            text += f"{bos_token}{B_INST} {B_SYS} {sys_prompt} {E_SYS} {prompt} {E_INST}"
        else:
            text += f"{bos_token}{B_INST} {prompt}{end_instr}"
        if res is not None:
            text += f" {res} {eos_token} "
    if len(history) == 0 or text.strip() == '':
        text = f"{bos_token}{B_INST} {B_SYS} {sys_prompt} {E_SYS} {message} {E_INST}"
    else:
        text += f"{bos_token}{B_INST} {message} {E_INST}"
    return text


def chat_multiturn_seq_format_v2(message, history):
    text = f"<|im_start|>system \nYou are a helpful assistant.</s><|im_start|>user \nHello world</s><|im_start|>assistant \nHi there, how can I help you today?</s><|im_start|>user \n{message}</s>"
    # response = "<|im_start|>assistant \nHi there, how can I help?</s>"
    return text

def parse_seallm_v1(output_):
    output_ = output_.split("[/INST]")[-1].replace("</s>", "").strip()
    return output_

def parse_seallm_v2(output_, split_str="Two rows should be returned in English, the first with the summary and the second with the short title."):
    output = output_.split(split_str)[-1].strip()
    # output = output_.strip()
    # if output.endswith("</s>"):
    #     output = output.split("</s>")[-2]
    # else:
    #     output = output.split("</s>")[-1]
    # output = output.strip()
    return output


def get_atep_prompt(cmt_list):
  templates_all = [
    "The aspect term of feedback refers to the primary elements that individuals share a viewpoint on.",
    "The aspect term in a remark signifies the key facets that people articulate their thoughts about.",
    "The aspect term from a comment pinpoints the main areas that people voice their opinion on.",
    "The aspect term of a commentary indicates the chief components that individuals express their judgment on.",
    "The aspect term in a note represents the main features that people share their perspective on.",
    "The aspect term associated with a comment identifies the primary factors that individuals provide their opinion on.",
    "The aspect term in a response denotes the primary dimensions that people express their views on.",
    "The aspect term in a statement signifies the main issues that people offer their opinion on.",
    "The aspect term of an observation refers to the essential aspects that individuals express their sentiment on.",
    "The aspect term in a suggestion indicates the primary points that people give their opinion on.",
    "The aspect term from a review signifies the key parts that people convey their thoughts on.",
    "The aspect term of a message refers to the main elements that individuals express their viewpoint on.",
    "The aspect term in a critique represents the main characteristics that people voice their opinion on.",
    "The aspect term associated with a feedback pinpoints the central matters that individuals offer their judgment on.",
    "The aspect term of an evaluation denotes the primary parameters that people articulate their views on.",
    "The aspect term in a discussion indicates the main themes that people express their opinion on.",
    "The aspect term from a remark refers to the key sectors that individuals share their perspective on.",
    "The aspect term of a comment outlines the main aspects that people provide their sentiment on.",
    "The aspect term in an opinion signifies the chief points that individuals express their thoughts on.",
    "The aspect term from an annotation indicates the central themes that people share their viewpoint on.",
    "The aspect term in an analysis represents the primary elements that individuals voice their opinion on.",
    "The aspect term of a statement pinpoints the main areas that people offer their judgment on.",
    "The aspect term associated with a note identifies the key factors that individuals articulate their views on.",
    "The aspect term in a feedback refers to the primary dimensions that people express their opinion on.",
    "The aspect term from a critique denotes the main issues that individuals express their perspective on.",
    "The aspect term of a response signifies the key parts that people give their thoughts on.",
    "The aspect term in a commentary refers to the essential aspects that individuals share their sentiment on.",
    "The aspect term associated with a review indicates the main features that people voice their viewpoint on.",
    "The aspect term from a suggestion represents the primary factors that individuals offer their opinion on.",
    "The aspect term in a message denotes the central matters that people express their judgment on."
  ]
  prompt_all = list()
  for item in cmt_list:
      prompt = f"{random.choice(templates_all)} If no opinion is expressed, the aspect terms should be 'NA'. Annotate the aspect terms of the following comment: '{item}'"
      prompt_all.append(prompt)
  return prompt_all

def split_sentences(text):
    """Splits text into sentences, handling multiple languages.
    """
    try:
        lang_code = detect(text) 
        if lang_code.startswith("zh"):
            doc = nlp_zh(text)
        elif lang_code == 'en':
            doc = nlp_en(text)
        else:
            sentences = split_sentences_ms_id(text)
            return sentences
    except:
        return [text]
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences

def split_sentences_ms_id(text):
    # Define patterns for sentence-ending punctuation followed by a space and a capital letter
    sentence_endings = r'(?<=[.!?])\s+(?=[A-Z])'

    # Handle common abbreviations and titles that might be misinterpreted as sentence endings
    abbreviations = r'\b(Prof|Dr|Mr|Ms|Mrs|Tn|Pn|Encik|Cik|Sdr|Sdri|Hj|Hjh|D|Dra)\.\s'

    # Use re.sub to prevent splitting at abbreviations
    text = re.sub(abbreviations, lambda x: x.group().replace('. ', '.#'), text)

    # Split the text into sentences
    sentences = re.split(sentence_endings, text)

    # Replace the placeholder back to the actual punctuation and remove extra spaces
    sentences = [s.replace('.#', '.').strip() for s in sentences]

    return sentences