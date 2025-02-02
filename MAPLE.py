import google.generativeai as genai
from google.generativeai.types import ContentType
from get_task import get_task, proces_input, proces_pseudoinput, zeroshot_prompt, process_labelembed, process_pseudoembed
from sentence_transformers import SentenceTransformer
from utils import *
import random, argparse
from tqdm import tqdm
import networkx as nx
import numpy as np
import json,os
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from process_bank77 import select_banking77_data, calculate_banking77_acc
from process_goemotion import select_goemo_data, calculate_goemo_acc
from process_bbh import calculate_bhh_acc
from process_gpqa import calculate_gpqa_acc
from process_fp import calculate_fp_acc
from google.generativeai.types import RequestOptions
from google.api_core import retry
from google.api_core.exceptions import RetryError, ResourceExhausted

parser = argparse.ArgumentParser(description="augmentation .")
parser.add_argument('-t', '--task', type=str, default='xsum', choices=['xsum', 'goemo', 'banking77', 'date','salient','tracking','gpqa', 'fp'],help='Task name')
parser.add_argument('-m', '--method', type=str, default='random', choices=['random', 'rag','rag_adpt', 'maple', 'zero', 'fewshot'],
                    help='method name')
parser.add_argument('-nl', '--labeled_num', type=int, default=20, help='number of labeled examples')
parser.add_argument('-nul', '--unlabeled_num', type=int, default=20, help='number of unlabeled examples')
parser.add_argument('-d', '--degree', type=int, default=20, help='degree of data graph')
parser.add_argument('-s', '--seed', type=int, default=0, help='random seed')
parser.add_argument('--save', action='store_true', help='save prediction')
args = parser.parse_args()
set_seed(args.seed)


##sample labeled data
def select_labeled(dataset, k):
    selected_indices = np.random.choice(len(dataset), k, replace=False)
    return selected_indices


def generate_embedd(dataset):
    tokenizer = AutoTokenizer.from_pretrained('facebook/contriever-msmarco')
    model = AutoModel.from_pretrained('facebook/contriever-msmarco')
    inputs = tokenizer(dataset, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)

    def mean_pooling(token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    embeddings = mean_pooling(outputs[0], inputs['attention_mask']).detach()
    return embeddings


def add_nodes(embeddings, G, emd, top_k=10):
    graph = G.copy()
    new_idx = graph.number_of_nodes()
    sim_scores = (emd@ embeddings.T).numpy()
    sim_scores = np.ravel(sim_scores)
    sorted_indices = np.argsort(sim_scores)[-top_k:]
    for idx in sorted_indices:
        if idx != new_idx:
            graph.add_edge(new_idx, idx)
    return graph


def select_unlabeled_maple(graph, node_set, k):
    remain_nodes = np.delete(np.arange(len(graph)), node_set)
    influences = []
    avg_d = 2 * graph.number_of_edges() / graph.number_of_nodes()
    for idx in remain_nodes:
        lengths = [nx.shortest_path_length(graph, source=idx, target=i) for i in node_set]
        num_paths = [np.log(len(list(nx.all_shortest_paths(graph, source=idx, target=i)))) for i in node_set]
        influence = np.mean(num_paths) - np.mean(lengths) * np.log(avg_d)
        influences.append(influence)
    sim_scores = np.ravel(np.array(influences))
    top_k_indices = np.argsort(sim_scores)[-k:]
    selected_indices = remain_nodes[top_k_indices]
    return selected_indices


def avoid_resourceerror(model, prompt):
    import time, google
    try:
        response = model.generate_content(prompt)
        return response
    except (google.api_core.exceptions.ResourceExhausted,) as e:
        print(f"Rate limit error: {e}. Retrying after 20 seconds...")
        time.sleep(20)
        return avoid_resourceerror(model, prompt)


if __name__ == "__main__":
    set_seed(args.seed)

    GOOGLE_API_KEY = 'YOUR_API_KEYS_HERE'
    genai.configure(api_key=GOOGLE_API_KEY)
    trainset, testset = get_task(args.task)

    if args.task == 'banking77':
        trainset = select_banking77_data(trainset, min(1000,len(trainset)), seed=args.seed)
        testset = select_banking77_data(testset, min(300,len(testset)), seed=args.seed)
    elif args.task == 'goemo':
        trainset = select_goemo_data(trainset, min(1000,len(trainset)), seed=args.seed)
        testset = select_goemo_data(testset, min(300,len(testset)), seed=args.seed)
    elif args.task in ['date','salient','tracking', 'gpqa']:
        sampled_idxs = np.random.choice(len(trainset), min(1000,len(trainset)), replace=False)
        trainset = [trainset[i] for i in sampled_idxs]
    else:
        sampled_idxs = np.random.choice(len(trainset), min(1000, len(trainset)), replace=False)
        trainset = [trainset[i] for i in sampled_idxs]
        test_sampled_idxs = np.random.choice(len(testset), min(300,len(testset)), replace=False)
        testset = [testset[i] for i in test_sampled_idxs]

    labeled_idxs = select_labeled(trainset, args.labeled_num)
    unlabeled_idxs = []
    remain_idxs = np.delete(np.arange(len(trainset)), labeled_idxs)
    if args.method == 'random':
        unlabeled_idxs = np.random.choice(remain_idxs, args.unlabeled_num, replace=False)
    elif args.method == 'maple':
        train_set = [process_labelembed(args.task, tem, labeled=False) for tem in trainset]
        embeddings = generate_embedd(train_set)
        G = nx.Graph()
        bar = tqdm(range(len(train_set)), desc=f'construct graph')
        for i in range(len(train_set)):
            sim_scores = (embeddings[i] @ embeddings.T).numpy()
            sim_scores = np.ravel(sim_scores)
            sorted_indices = np.argsort(sim_scores)[-args.degree - 1:-1]
            for idx in sorted_indices:
                if idx != i:
                    G.add_edge(i, idx)
            bar.update(1)

        #make sure the graph is connected

        if not nx.is_connected(G):
            components = list(nx.connected_components(G))
            for i in range(len(components) - 1):
                node1 = next(iter(components[i]))
                node2 = next(iter(components[i + 1]))
                G.add_edge(node1, node2)
        unlabeled_idxs = select_unlabeled_maple(G, labeled_idxs, args.unlabeled_num)
        test_set = [process_labelembed(args.task, tem, labeled=False) for tem in testset]
        test_embeddins = generate_embedd(test_set)


    elif 'rag' in args.method:
        remain_set = [process_labelembed(args.task, trainset[id], labeled=False) for id in remain_idxs]
        test_set = [process_labelembed(args.task, tem, labeled=False) for tem in testset]
        remain_embeddings = generate_embedd(remain_set)
        test_embeddins = generate_embedd(test_set)
        average_embedding = test_embeddins.mean(dim=0, keepdim=True)
        sim_scores = (average_embedding @ remain_embeddings.T).numpy()
        sim_scores=np.ravel(sim_scores)
        top_k_indices = np.argsort(sim_scores)[-args.unlabeled_num:]
        unlabeled_idxs = remain_idxs[top_k_indices]

    else:
        print("method not implemented")

    labeled_demos = [trainset[idx] for idx in labeled_idxs]
    generation_config = {"temperature": 0}
    model = genai.GenerativeModel('gemini-1.5-flash-002', generation_config=generation_config)
    eval_scores = []
    feeds = []
    pseudo_data = []

    if (args.unlabeled_num>0) and (args.method not in ['fewshot','zero']):
        for unlabel_id in unlabeled_idxs:
            tem = trainset[unlabel_id]

            prompt = proces_input(args.task, labeled_demos, [], tem)
            response = avoid_resourceerror(model, prompt)
            pseudo_data.append({'example':tem, 'pred':response.text})



        if args.method == 'maple':
            labeled_forembed = [process_labelembed(args.task, tem, labeled=True) for tem in labeled_demos]
            pseudolabeled_forembed = [process_pseudoembed(args.task, tem) for tem in pseudo_data]
            mixed_data = labeled_forembed+pseudolabeled_forembed
            labeled_embeddings = generate_embedd(mixed_data)
            G_new = nx.Graph()
            for i in range(len(mixed_data)):
                sim_scores = (labeled_embeddings[i] @ labeled_embeddings.T).numpy()
                sim_scores = np.ravel(sim_scores)
                sorted_indices = np.argsort(sim_scores)[-10 - 1:-1]
                for idx in sorted_indices:
                    if idx != i:
                        G_new.add_edge(i, idx)

            if not nx.is_connected(G_new):
                components = list(nx.connected_components(G_new))
                for i in range(len(components) - 1):
                    node1 = next(iter(components[i]))
                    node2 = next(iter(components[i + 1]))
                    G_new.add_edge(node1, node2)


            mixed_data = [process_labelembed(args.task, tem, labeled=False) for tem in labeled_demos] + [process_pseudoembed(args.task, tem, labeled=False) for tem in pseudo_data]
            embed_forq = generate_embedd(mixed_data)


        if args.save:
            pseudo_data_dir = (f'pseudo_data/{args.task}')
            if not os.path.exists(pseudo_data_dir):
                os.makedirs(pseudo_data_dir)
            pseudo_data_path = f'pseudo_data/{args.task}/{args.method}_nl{args.labeled_num}_nul{args.unlabeled_num}_seed{args.seed}.json'
            with open(pseudo_data_path, mode='w', encoding='utf-8') as f:
                f.write(json.dumps(pseudo_data, indent=2))


    def process_batch(testset_batch, args, model, labeled_demos, pseudo_data, G_new, labeled_embeddings):
        prompts = []
        for tem in testset_batch:
            if args.method == 'zero':
                prompts.append(zeroshot_prompt(args.task, tem))
            elif args.method == 'fewshot':
                prompts.append(proces_input(args.task, labeled_demos, [], tem))
            elif args.method == 'maple':
                testnidx = G_new.number_of_nodes()
                avg_d = 2 * G_new.number_of_edges() / G_new.number_of_nodes()
                query = process_labelembed(args.task, tem, labeled=False)
                query_embeddins = generate_embedd(query)
                G_new_withtest = add_nodes(labeled_embeddings, G_new, query_embeddins, 10)
                influences = []

                for idx in range(len(mixed_data)):
                    totestlen = nx.shortest_path_length(G_new_withtest, source=idx, target=testnidx)
                    totestnum_paths = len(list(nx.all_shortest_paths(G_new_withtest, source=idx, target=testnidx)))
                    influence = totestnum_paths - totestlen * np.log(avg_d)
                    influences.append(influence)

                selected_indices = np.where(np.array(influences) > np.percentile(influences, 25))[0]

                labeled_demos_filtered = []
                pseudolabeled_demos_filtered = []

                for t in selected_indices:
                    if t < len(labeled_demos):
                        labeled_demos_filtered.append(labeled_demos[t])
                    else:
                        pseudolabeled_demos_filtered.append(pseudo_data[t - len(labeled_demos)])

                prompts.append(proces_pseudoinput(args.task, labeled_demos_filtered, pseudolabeled_demos_filtered, tem))

            elif args.method == 'rag_adpt':
                query = process_labelembed(args.task, tem, labeled=False)
                query_embeddins = generate_embedd(query)
                sim_scores = (query_embeddins @ labeled_embeddings.T).numpy()
                influences = np.ravel(sim_scores)

                selected_indices = np.where(influences > np.percentile(influences, 25))[0]

                labeled_demos_filtered = []
                pseudolabeled_demos_filtered = []

                for t in selected_indices:
                    if t < len(labeled_demos):
                        labeled_demos_filtered.append(labeled_demos[t])
                    else:
                        pseudolabeled_demos_filtered.append(pseudo_data[t - len(labeled_demos)])

                prompts.append(proces_pseudoinput(args.task, labeled_demos_filtered, pseudolabeled_demos_filtered, tem))
            else:
                prompts.append(proces_pseudoinput(args.task, labeled_demos, pseudo_data, tem))

        responses = [avoid_resourceerror(model, prompt) for prompt in prompts]

        eval_scores = []
        feeds = []

        for tem, response in zip(testset_batch, responses):
            if args.task == 'xsum':
                accurate = compute_rougeL(tem['summary'], response.text)
            elif args.task == 'banking77':
                accurate = calculate_banking77_acc(response.text, tem)
            elif args.task == 'goemo':
                accurate = calculate_goemo_acc(response.text, tem)
            elif args.task == 'gpqa':
                accurate = calculate_gpqa_acc(response.text, tem)
            elif args.task == 'fp':
                accurate = calculate_fp_acc(response.text, tem)
            elif args.task in ['date', 'salient', 'tracking']:
                accurate = calculate_bhh_acc(response.text, tem)
            else:
                print('unknown task')
                accurate = None

            if accurate is not None:
                eval_scores.append(float(accurate))

            output_dict = {'query': tem, 'pred': response.text}
            feeds.append(output_dict)

        return eval_scores, feeds


    batch_size = 16
    eval_scores = []
    feeds = []

    if args.method != 'maple':
        G_new = nx.Graph()
        embed_forq = []

    for i in range(0, len(testset), batch_size):
        batch = testset[i:i + batch_size]
        batch_scores, batch_feeds = process_batch(batch, args, model, labeled_demos, pseudo_data, G_new,
                                                  embed_forq)
        eval_scores.extend(batch_scores)
        feeds.extend(batch_feeds)

    # Save results
    if args.save:
        output_file = f'{args.task}_pseudolabeled_result/{args.method}_nl{args.labeled_num}_nul{args.unlabeled_num}_seed{args.seed}.json'
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, mode='w', encoding='utf-8') as feedsjson:
            feedsjson.write(json.dumps(feeds, indent=2))

    result_summary = np.mean(eval_scores)
    print(result_summary)