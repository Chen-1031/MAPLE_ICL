from datasets import load_dataset
import random

all_labels = ["admiration",
                "amusement",
                "anger",
                "annoyance",
                "approval",
                "caring",
                "confusion",
                "curiosity",
                "desire",
                "disappointment",
                "disapproval",
                "disgust",
                "embarrassment",
                "excitement",
                "fear",
                "gratitude",
                "grief",
                "joy",
                "love",
                "nervousness",
                "optimism",
                "pride",
                "realization",
                "relief",
                "remorse",
                "sadness",
                "surprise",
                "neutral"]


def select_goemo_data(given_dataset, total_count, seed=0):
    random.seed(seed)
    label_to_data_dict = {}
    for data in given_dataset:
        if len(data['labels']) == 1:
            cur_label = data['labels'][0]
            if cur_label in label_to_data_dict:
                label_to_data_dict[cur_label].append(data)
            else:
                label_to_data_dict[cur_label] = [data]

    for key in label_to_data_dict:
        random.shuffle(label_to_data_dict[key])

    selected_data_list = []
    data_label_list = list(label_to_data_dict.keys())
    selected_label_to_count = {key: 0 for key in data_label_list}

    while len(selected_data_list) < total_count:
        for key in data_label_list:
            if len(selected_data_list) >= total_count:
                break
            if selected_label_to_count[key] < len(label_to_data_dict[key]):
                selected_data_list.append(label_to_data_dict[key][selected_label_to_count[key]])
                selected_label_to_count[key] += 1
        if all(selected_label_to_count[key] >= len(label_to_data_dict[key]) for key in data_label_list):
            break

    return selected_data_list

def format_goemo_embed(data):
    return "comment: " + data['text'] + "\nemotion category: " + all_labels[data['labels'][0]]

def format_zerogoemo_prompt(test):

    prompt =  "I am going to provide a comment and I want you to predict the emotion category of the comment. Give only the emotion category, and no extra commentary, formatting, or chattiness."
    prompt = prompt + 'You can only make prediction from the following categories: '
    for i, word in enumerate(all_labels):
        if i != len(all_labels) - 1:
            prompt = prompt + word + ', '
        else:
            prompt = prompt + word + '.\n'
    prompt = prompt + "comment: " + test['text'] + "\nemotion category: "

    return prompt

def format_goemo_prompt(labeled, unlabled, test):

    prompt = 'Given a comment, please predict the emotion category of this comment. Here are several examples.\n'
    for data in labeled:
        prompt = prompt + "comment: " + data['text'] + "\nemotion category: " + all_labels[data['labels'][0]] + '\n'
    if len(unlabled) > 0:
        prompt = "Here are several examples of comment without the ground truth emotion category. \n"
        for data in unlabled:
            prompt = prompt + "comment: " + data['text'] + '\n'

    #prompt = prompt + 'Given a customer service query, please predict the intent of the query. The predict answer must come from the demonstration examples with the exact format.'
    prompt = prompt + "I am going to provide another comment and I want you to predict the emotion category of the comment. Give only the emotion category, and no extra commentary, formatting, or chattiness."
    prompt = prompt + 'You can only make prediction from the following categories: '
    for i, word in enumerate(all_labels):
        if i != len(all_labels) - 1:
            prompt = prompt + word + ', '
        else:
            prompt = prompt + word + '.\n'
    prompt = prompt + "comment: " + test['text'] + "\nemotion category: "

    return prompt

def format_pseudogoemo_prompt(labeled, pseudolabled, test):

    prompt = 'Given a comment, please predict the emotion category of this comment. Here are several examples.\n'
    for data in labeled:
        prompt = prompt + "comment: " + data['text'] + "\nemotion category: " + all_labels[data['labels'][0]] + '\n'
    if len(pseudolabled) > 0:
        for data in pseudolabled:
            prompt = prompt + "comment: " + data['example']['text'] + "\nemotion category: " + data['pred'] + '\n'

    #prompt = prompt + 'Given a customer service query, please predict the intent of the query. The predict answer must come from the demonstration examples with the exact format.'
    prompt = prompt + "I am going to provide another comment and I want you to predict the emotion category of the comment. Give only the emotion category, and no extra commentary, formatting, or chattiness."
    prompt = prompt + 'You can only make prediction from the following categories: '
    for i, word in enumerate(all_labels):
        if i != len(all_labels) - 1:
            prompt = prompt + word + ', '
        else:
            prompt = prompt + word + '.\n'
    prompt = prompt + "comment: " + test['text'] + "\nemotion category: "

    return prompt

def calculate_goemo_acc(response, tem):
    temp_prompt = "emotion category:"
    if tem['text'] in response:
        response = list(response.split(tem['text']))[-1].strip().split(temp_prompt)
        if len(response) > 1:
            response = response[1].split("comment: ")[0]
        else:
            response = response[0]
    else:
        response = response.split("comment: ")[0]
    response = response.strip().split("\n")[0]

    response = response.lower().strip()
    label = all_labels[tem['labels'][0]]
    label = label.lower()

    if response == label:
        return 1
    else:
        return 0



