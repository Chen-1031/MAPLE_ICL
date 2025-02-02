# MAPLE: Many-Shot Adaptive Pseudo-Labeling In-Context Learning

This repository contains the code for the paper MAPLE: Many-Shot Adaptive Pseudo-Labeling In-Context Learning.

In-Context Learning (ICL) empowers Large Language Models (LLMs) to tackle diverse tasks by incorporating multiple input-output examples, known as demonstrations, into the input of LLMs. More recently, advancements in the expanded context windows of LLMs have led to many-shot ICL, which uses hundreds of demonstrations and outperforms few-shot ICL, which relies on fewer examples. However, this approach is often hindered by the high cost of obtaining large amounts of labeled data. To address this challenge, we propose **M**any-Shot **A**daptive **P**seudo-**L**ab**E**ling, namely **MAPLE**, a novel influence-based many-shot ICL framework that utilizes pseudo-labeled samples to compensate for the lack of label information. 
We first identify a subset of impactful unlabeled samples and perform pseudo-labeling on them by querying LLMs. These pseudo-labeled samples are then adaptively selected and tailored to each test query as input to improve the performance of many-shot ICL, without significant labeling costs.
Extensive experiments on real-world datasets demonstrate the effectiveness of our framework, showcasing its ability to enhance LLM adaptability and performance with limited labeled data.


## Running

```
python msICL.py --task xsum --method maple --labeled_num 20 --unlabeled_num 20 --degree 20 --seed 0 
```

## Parameter Summary: The parameters used in `MAPLE.py` are summarized as follows:

| Parameter     | Description                                                                                      |
|---------------|--------------------------------------------------------------------------------------------------|
| task          | Dataset to use (default: xsum)                                                                   |
| method        | Method to selected samples (default: random)                                                     |
| labeled_num   | Number of labeled examples, i.e., size of $\mathcal{D}_L$ (default: 20)                          |
| unlabeled_num | Nnumber of unlabeled examples for pseudo-labeling, i.e., size of $\mathcal{D}^*_U$ (default: 20) |
| degree        | Degree of the constructed graph $\mathcal{G}$ (default: 20)                                      |
| seed          | Random seed (default: 0)                                                                         |


