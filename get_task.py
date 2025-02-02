import json, os
import pandas as pd
from datasets import load_dataset
from process_bank77 import format_banking77_prompt, format_pseudobanking77_prompt, format_zerobanking77_prompt, format_banking77_embed, format_pseudobanking77_prompt_reverse
from process_goemotion import format_goemo_prompt, format_pseudogoemo_prompt, format_zerogoemo_prompt, format_goemo_embed
from process_bbh import transform_data, format_bbh_prompt, format_pseudobbh_prompt, format_pseudobbh_prompt_reverse
from process_gpqa import format_gpqa_prompt, load_gpqa_examples, format_pseudogpqa_prompt, format_pseudogpqa_prompt_reverse
from process_fp import load_fp_data, format_fp_prompt, format_pseudofp_prompt

def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]

def get_task(dataset_name):
    if not os.path.exists('data'):
        os.makedirs('data')
    if dataset_name == 'gsm8k':
        trainset = read_jsonl("data/gsm8k/train.jsonl")
        testset = read_jsonl("data/gsm8k/test.jsonl")

    elif dataset_name == 'xsum':
        xsum_dataset = load_dataset('xsum', cache_dir='data/', trust_remote_code=True)
        trainset = [e for e in xsum_dataset['train']]
        testset = [e for e in xsum_dataset['test']]

    elif dataset_name == 'banking77':
        dataset = load_dataset("banking77", cache_dir='data/', trust_remote_code=True)
        trainset = dataset['train']
        testset = dataset['test']

    elif dataset_name == 'goemo':
        dataset = load_dataset("go_emotions", cache_dir='data/', trust_remote_code=True)
        trainset = dataset['train']
        testset = dataset['test']

    elif dataset_name in ['date','salient','tracking']:
        trainpath = f"data/{dataset_name}_train.json"
        if not os.path.exists(trainpath):
            with open(f"data/{dataset_name}_origin.json", 'rb') as fh:
                datapool = json.load(fh)
            transform_data(datapool, dataset_name)
        with open(trainpath, 'rb') as fh:
            trainset = json.load(fh)
        testpath = f"data/{dataset_name}_eval.json"
        with open(testpath, 'rb') as fh:
            testset = json.load(fh)

        testset = testset['examples']

    elif dataset_name == "gpqa":
        testdf = pd.read_csv("data/dataset/gpqa_diamond.csv")
        orgin_data = pd.read_csv("data/dataset/gpqa_main.csv")
        traindf = orgin_data[~orgin_data['Record ID'].isin(testdf['Record ID'])]
        trainset = load_gpqa_examples(traindf)
        testset = load_gpqa_examples(testdf)

    elif dataset_name == "fp":
        filepath = "data/FinancialPhraseBank-v1.0/Sentences_75Agree.txt"
        trainset, testset = load_fp_data(filepath)

    return trainset, testset

def process_pseudoembed(dataset_name, tem, labeled=True):
    if dataset_name == 'gsm8k':
        if labeled:
            for_embed = "Question: " + tem['example']['question'] + "\nAnswer: " + tem['pred']
        else:
            for_embed = "Question: " + tem['example']['question'] + "\nAnswer: "
    elif dataset_name == 'xsum':
        if labeled:
            for_embed = "Article: " + tem['example']['document'] + "\nSummary: " + tem['pred']
        else:
            for_embed = "Article: " + tem['example']['document'] + "\nSummary: "
    elif dataset_name == 'fp':
        if labeled:
            for_embed = "Sentence: " + tem['example']['sentence'] + "\nAnswer: " + tem['pred']
        else:
            for_embed = "Sentence: " + tem['example']['sentence'] + "\nAnswer: "
    elif dataset_name == 'banking77':
        if labeled:
            for_embed = "service query: " + tem['example']['text'] + "\nintent category: "
        else:
            for_embed = "service query: " + tem['example']['text'] + "\nintent category: "
    elif dataset_name == 'goemo':
        if labeled:
            for_embed = "comment: " + tem['example']['text'] + "\nemotion category: "
        else:
            for_embed = "comment: " + tem['example']['text'] + "\nemotion category: "
    elif dataset_name in ['date', 'salient', 'tracking', 'gpqa']:
        if labeled:
            for_embed = "Question: " + tem['example']['input'] + "\nAnswer: " + tem['pred']
        else:
            for_embed = "Question: " + tem['example']['input'] + "\nAnswer: "

    return for_embed

def process_labelembed(dataset_name, tem, labeled=False):
    if dataset_name == 'gsm8k':
        if labeled:
            for_embed = "Question: " + tem['question'] + "\nAnswer: " + tem['answer']
        else:
            for_embed = "Question: " + tem['question'] + "\nAnswer: "
    elif dataset_name == 'xsum':
        if labeled:
            for_embed = "Article: " + tem['document'] + "\nSummary: " + tem['summary']
        else:
            for_embed = "Article: " + tem['document'] + "\nSummary: "
    elif dataset_name == 'fp':
        if labeled:
            for_embed = "Sentence: " + tem['sentence'] + "\nAnswer: " + tem['label']
        else:
            for_embed = "Sentence: " + tem['sentence'] + "\nAnswer: "
    elif dataset_name == 'banking77':
        if labeled:
            for_embed = format_banking77_embed(tem)
        else:
            for_embed = "service query: " + tem['text'] + "\nintent category: "
    elif dataset_name == 'goemo':
        if labeled:
            for_embed = format_goemo_embed(tem)
        else:
            for_embed = "comment: " + tem['text'] + "\nemotion category: "
    elif dataset_name in ['date', 'salient', 'tracking', 'gpqa']:
        if labeled:
            for_embed = "Question: " + tem['input'] + "\nAnswer: " + tem['target']
        else:
            for_embed = "Question: " + tem['input'] + "\nAnswer: "

    return for_embed


def proces_input(dataset_name, labeled, unlabeled, test):

    if dataset_name == 'gsm8k':
        prompt = "Here are several examples. \n"
        for tem in labeled:
            prompt += "question: {} \nanswer: {}\n".format(tem['question'], tem['answer'])
        if len(unlabeled) > 0:
            prompt = "Here are several examples without answer. \n"
            for tem in unlabeled:
                prompt += "question: {} \n".format(tem['question'])
        prompt +=  "Please answer the following question. \n {} \n Explain your reasoning. Your final answer should be a single numerical number, in the form \boxed{{answer}}, at the end of your response.".format(
            test['question'])
    elif dataset_name == 'xsum':
        prompt = "You are an expert in article summarization. I am going to give you some examples of article and its summary in fluent English. Here are several examples. \n"
        for tem in labeled:
            prompt += "Article: {} \nSummary: {}\n".format(tem['document'], tem['summary'])
        if len(unlabeled) > 0:
            prompt = "Here are several examples without summary. \n"
            for tem in unlabeled:
                prompt += "Article: {} \n".format(tem['document'])
        prompt += "I am going to provide another article and I want you to summarize it. Give only the summary, and no extra commentary, formatting, or chattiness."
        prompt = prompt + "Article: " + test['document']
    elif dataset_name == 'banking77':
        prompt = format_banking77_prompt(labeled, unlabeled, test)
    elif dataset_name == 'goemo':
        prompt = format_goemo_prompt(labeled, unlabeled, test)
    elif dataset_name == 'gpqa':
        prompt = format_gpqa_prompt(labeled, unlabeled, test)
    elif dataset_name == 'fp':
        prompt = format_fp_prompt(labeled, unlabeled, test)
    elif dataset_name in ['date','salient','tracking']:
        prompt = format_bbh_prompt(labeled, unlabeled, test)

    return prompt



def zeroshot_prompt(dataset_name, test):
    if dataset_name == 'gsm8k':
        prompt =  "Please answer the following question. \n {} \n Explain your reasoning. Your final answer should be a single numerical number, in the form \boxed{{answer}}, at the end of your response.".format(
            test['question'])
    elif dataset_name == 'xsum':
        prompt = "You are an expert in article summarization. I am going to provide an article and I want you to summarize it. Give only the summary, and no extra commentary, formatting, or chattiness."
        prompt = prompt + "Article: " + test['document']
    elif dataset_name == 'banking77':
        prompt = format_zerobanking77_prompt(test)
    elif dataset_name == 'goemo':
        prompt = format_zerogoemo_prompt(test)
    elif dataset_name == 'gpqa':
        prompt = 'You are an expert in multiple-choice question answering tasks.'
        prompt = prompt + "I am going to provide another question and I want you to give its answer. Give only the choice the correct answer by selecting one of the options (e.g., '(A)', '(B)')."
        prompt = prompt + "Question: " + test['input']
    elif dataset_name == 'fp':
        prompt = 'You are an expert in financial sentiment analysis.'
        prompt = prompt + "I am going to provide a sentence and I want you to analyze the sentiment of it and respond with only one word: 'positive', 'negative', or 'neutral'. No extra commentary, formatting, or chattiness."
        prompt = prompt + "Sentence: " + test['sentence']
    elif dataset_name in ['date','salient','tracking']:
        prompt = 'You are an expert in multiple-choice question answering tasks.'
        prompt = prompt + "I am going to provide another question and I want you to give its answer. Give only the choice the correct answer by selecting one of the options (e.g., '(A)', '(B)')."
        prompt = prompt + "Question: " + test['input']
    return prompt

def proces_pseudoinput(dataset_name, labeled, pseudolabled, test):
    if dataset_name == 'gsm8k':
        prompt = "Here are several examples. \n"
        for tem in labeled:
            prompt += "question: {} \nanswer: {}\n".format(tem['question'], tem['answer'])
        if len(pseudolabled) > 0:
            for tem in pseudolabled:
                prompt += "question: {} \nanswer: {}\n".format(tem['example']['question'], tem['pred'])
        prompt +=  "Please answer the following question. \n {} \n Explain your reasoning. Your final answer should be a single numerical number, in the form \boxed{{answer}}, at the end of your response.".format(
            test['question'])
    elif dataset_name == 'xsum':
        prompt = "You are an expert in article summarization. I am going to give you some examples of article and its summary in fluent English. Here are several examples. \n"
        for tem in labeled:
            prompt += "Article: {} \nSummary: {}\n".format(tem['document'], tem['summary'])
        if len(pseudolabled) > 0:
            for tem in pseudolabled:
                prompt += "Article: {} \nSummary: {}\n".format(tem['example']['document'], tem['pred'])
        prompt += "I am going to provide another article and I want you to summarize it. Give only the summary, and no extra commentary, formatting, or chattiness."
        prompt = prompt + "Article: " + test['document']
    elif dataset_name == 'banking77':
        prompt = format_pseudobanking77_prompt(labeled, pseudolabled, test)
    elif dataset_name == 'goemo':
        prompt = format_pseudogoemo_prompt(labeled, pseudolabled, test)
    elif dataset_name == 'gpqa':
        prompt = format_pseudogpqa_prompt(labeled, pseudolabled, test)
    elif dataset_name == 'fp':
        prompt = format_pseudofp_prompt(labeled, pseudolabled, test)
    elif dataset_name in ['date','salient','tracking']:
        prompt = format_pseudobbh_prompt(labeled, pseudolabled, test)

    return prompt





