import numpy as np
import pandas as pd
from collections import namedtuple
import random
import json, re
import unicodedata
import collections
import string
from typing import Any

#Example = namedtuple('Example', ['question', 'choice1', 'choice2', 'choice3', 'choice4', 'correct_index'])

def load_gpqa_examples(question_df):
    letter_answer_choices = ['(A)', '(B)', '(C)', '(D)']
    random.seed(42)
    def shuffle_choices_and_create_example(row):
        list_choices = [row['Incorrect Answer 1'], row['Incorrect Answer 2'], row['Incorrect Answer 3'], row['Correct Answer']]
        random.shuffle(list_choices)
        prompt = f"{row.Question}"
        prompt += f"\n\nChoices:\n(A) {list_choices[0]}\n(B) {list_choices[1]}\n(C) {list_choices[2]}\n(D) {list_choices[3]}"
        example = {
            "input": prompt,
            "target": f"{letter_answer_choices[list_choices.index(row['Correct Answer'])]}"
        }
        return example

    return [shuffle_choices_and_create_example(row) for _, row in question_df.iterrows()]


def format_gpqa_prompt(labeled, unlabled, test):

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

def format_pseudogpqa_prompt(labeled, pseudolabled, test):

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


def calculate_gpqa_acc(response, tem):

    pred_answer = normalize_answer(response)

    gt=normalize_answer(tem['target'])

    if not pred_answer:
        return 0
    else:
        return compute_subspan_em(gt, pred_answer)
