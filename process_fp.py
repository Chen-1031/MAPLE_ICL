import random, re

def load_fp_data(filepath):
    data=[]
    with open(filepath, encoding="iso-8859-1") as f:
        for id_, line in enumerate(f):
            sentence, label = line.rsplit("@", 1)
            data.append({"sentence": sentence, "label": label})
    random.seed(42)
    test_data = random.sample(data, 500)
    train_data = [item for item in data if item not in test_data]
    return train_data, test_data

def format_fp_prompt(labeled, unlabled, test):

    prompt = 'You are an expert in financial sentiment analysis. Here are several examples.\n'
    for data in labeled:
        prompt = prompt + "Sentence: " + data['sentence'] + "\nAnswer: " + data['label'] + '\n'
    if len(unlabled) > 0:
        prompt = "Here are several examples of sentiment analysis without the ground truth answer. \n"
        for data in unlabled:
            prompt = prompt + "Sentence: " + data['sentence'] + '\n'

    prompt = prompt + "I am going to provide another sentence and I want you to analyze the sentiment of it and respond with only one word: 'positive', 'negative', or 'neutral'. No extra commentary, formatting, or chattiness."
    prompt = prompt + "Sentence: " + test['sentence']

    return prompt

def format_pseudofp_prompt(labeled, pseudolabled, test):

    prompt = 'You are an expert in financial sentiment analysis. Here are several examples.\n'
    for data in labeled:
        prompt = prompt + "Sentence: " + data['sentence'] + "\nAnswer: " + data['label'] + '\n'
    if len(pseudolabled) > 0:
        for data in pseudolabled:
            prompt = prompt + "Sentence: " + data['example']['sentence'] + "\nAnswer: " + data['pred'] + '\n'

    prompt = prompt + "I am going to provide another sentence and I want you to analyze the sentiment of it and respond with only one word: 'positive', 'negative', or 'neutral'. No extra commentary, formatting, or chattiness."
    prompt = prompt + "Sentence: " + test['sentence']

    return prompt


def calculate_fp_acc(response, tem):
    cleaned_label = re.sub(r'\s+', '', tem['label'])
    cleaned_res = re.sub(r'\s+', '', response)

    if not cleaned_res:
        return 0
    else:
        return 1.0 if cleaned_label in cleaned_res else 0.0