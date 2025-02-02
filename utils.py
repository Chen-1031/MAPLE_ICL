import os, re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch
import random
import numpy as np
from rouge import Rouge

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def calculate_sentence_transformer_embedding(text_to_encode,emb_model):
    num = len(text_to_encode)
    embeddings = []
    bar = tqdm(range(0,num,20),desc='calculate embeddings')
    for i in range(0,num,20):
        embeddings += emb_model.encode(text_to_encode[i:i+20]).tolist()
        bar.update(1)
    embeddings = torch.tensor(embeddings)
    mean_embeddings = torch.mean(embeddings, 0, True)
    embeddings = embeddings - mean_embeddings
    return embeddings

# def parse_answer(input_str):
#     pattern = r"\{([0-9.,$]*)\}"
#     matches = re.findall(pattern, input_str)
#
#     solution = None
#
#     for match_str in matches[::-1]:
#         solution = re.sub(r"[^0-9.]", "", match_str)
#         if solution:
#             break
#
#     return solution

def solve_math_problems(input_str):
    pattern = r"\d+\.?\d*"
    matches = re.findall(pattern, input_str)
    if matches:
        return matches[-1]
    return None
def compute_accuracy(gt, pred_solution):
    answers = solve_math_problems(gt)

    if answers is None:
        return None

    pattern = r"\boxed{([^}]*)}"
    match = re.search(pattern, pred_solution)

    if match:
        pred_answer = match.group(1)

    # pred_answer = parse_answer(pred_solution)
    if match is None:
        pred_answer = solve_math_problems(pred_solution)

    if pred_answer is None:
        return 1

    if float(answers) == float(pred_answer):
        return 1
    else:
        return 0

def compute_rougeL(gt, pred_solution):

    rouge = Rouge()
    scores = rouge.get_scores(pred_solution, gt)
    return scores[0]['rouge-l']['f']

# rouge = Rouge()
# scores = rouge.get_scores('The quick brown fox jumps over the lazy dog',
#                       'The quick brown dog jumps on the log.')
# scores[0]['rouge-l']
