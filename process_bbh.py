import json, re
import unicodedata
import collections
import string
from typing import Any

def transform_data(datapool,dataname):
    processed=[]
    for data in datapool['examples']:
        input_text = data["input"]
        target_scores = data["target_scores"]
        options = list(target_scores.keys())
        if len(options) == 6:
            labels = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)"]
        else:
            labels = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)"]
        options_str = "\nOptions:\n" + "\n".join([f"{labels[i]} {options[i]}" for i in range(len(options))])
        correct_index = options.index([key for key, value in target_scores.items() if value == 1][0])
        correct_label = labels[correct_index]
        output = {
            "input": f"{input_text}{options_str}",
            "target": f"{correct_label}"
        }
        processed.append(output)

    with open(f'data/{dataname}_train.json', 'w') as file:
        json.dump(processed, file)

def format_bbh_prompt(labeled, unlabled, test):

    prompt = 'You are an expert in multiple-choice question answering tasks. I am going to give you some examples in a multiple-choice question answering format. Here are several examples.\n'
    for data in labeled:
        prompt = prompt + "Question: " + data['input'] + "\nAnswer: " + data['target'] + '\n'
    if len(unlabled) > 0:
        prompt = "Here are several examples of multiple-choice questions without the ground truth answer. \n"
        for data in unlabled:
            prompt = prompt + "Question: " + data['input'] + '\n'

    prompt = prompt + "I am going to provide another question and I want you to predict its answer. Give only the choice the correct answer by selecting one of the options (e.g., '(A)', '(B)')."
    prompt = prompt + "Question: " + test['input']

    return prompt

def format_pseudobbh_prompt(labeled, pseudolabled, test):

    prompt = 'You are an expert in multiple-choice question answering tasks. I am going to give you some examples in a multiple-choice question answering format. Here are several examples.\n'
    for data in labeled:
        prompt = prompt + "Question: " + data['input'] + "\nAnswer: " + data['target'] + '\n'
    if len(pseudolabled) > 0:
        for data in pseudolabled:
            prompt = prompt + "Question: " + data['example']['input'] + "\nAnswer: " + data['pred'] + '\n'

    prompt = prompt + "I am going to provide another question and I want you to predict its answer. Give only the choice the correct answer by selecting one of the options (e.g., '(A)', '(B)')."
    prompt = prompt + "Question: " + test['input']

    return prompt

def normalize_answer(s: str) -> str:
  """Taken from SQuAD evaluation."""

  s = unicodedata.normalize("NFD", s)

  def remove_articles(text: str) -> str:
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)

  def white_space_fix(text: str) -> str:
    return " ".join(text.split())

  def remove_punc(text: str) -> str:
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)

  def lower(text: str) -> str:
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_em(gt, pred_answer):
  """Calculates exact match score. Taken from SQuAD evaluation."""
  return float(gt == pred_answer)

def compute_subspan_em(gt, pred_answer) -> float:
  """Calculates subspan match score."""
  return 1.0 if gt in pred_answer else 0.0


def calculate_bhh_acc(response, tem):

    pred_answer = normalize_answer(response)

    gt=normalize_answer(tem['target'])

    if not pred_answer:
        return 0
    else:
        return compute_subspan_em(gt, pred_answer)