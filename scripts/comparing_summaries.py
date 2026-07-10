import json
import time

import numpy as np

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from genechat.models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer

#Transformer Modules
from transformers import AutoTokenizer, EsmModel, AutoModel
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

from typing import List

from scipy.spatial.distance import cosine, cdist

import matplotlib.pyplot as plt

"""
Vicuna-13b-v1.5

llama_model = "/data2/gene_chat/llama_pretrained/vicuna-13b-v1.5"

#Load the tokenizer
print("-->Loading tokenizer...")
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)

#Set the padding token to the eos token
llama_tokenizer.pad_token = llama_tokenizer.eos_token

#load the model
print("-->Loading model...")
llama_model = LlamaForCausalLM.from_pretrained(llama_model, torch_dtype=torch.float16, device_map='auto')
print(llama_model.device)

#Not going to train the model
for name, param in llama_model.named_parameters():
    param.requires_grad = False

#Print the number of trainig parameters - should be 0
llama_model.print_trainable_parameters()

#Set the model to eval mode
llama_model.eval()
"""

simcse_path = "princeton-nlp/sup-simcse-roberta-large"

tokenizer = AutoTokenizer.from_pretrained(simcse_path)
model = AutoModel.from_pretrained(simcse_path)

summary_file = "/data2/gene_chat/exon_count/data/train_temp_1/test_set/qa_summary_rule_2.json"
embeddings_file = "/data2/gene_chat/exon_count/data/train_temp_1/test_set/summary_embeddings.json"

with torch.no_grad():

    with open(summary_file, "r") as f:
        records = json.load(f)
    
    embeddings = []

    references = [record["Summary"] for record in records]
    candidates = references

    reference_embeddings = []
    candidate_embeddings = []

    for reference in references: 
        reference_tokens = tokenizer(reference, padding=True, truncation=True, return_tensors="pt")
        reference_embedding = model(**reference_tokens, output_hidden_states=True, return_dict=True).pooler_output
        reference_embeddings.append(reference_embedding.cpu().numpy())
        candidate_embeddings.append(reference_embedding.cpu().numpy())
    
    reference_embeddings = np.vstack(reference_embeddings)
    candidate_embeddings = np.vstack(candidate_embeddings)

    print(reference_embeddings.shape)
    print(candidate_embeddings.shape)

    cosine_similarities = 1 - cdist(reference_embeddings, candidate_embeddings, metric="cosine")

    np.fill_diagonal(cosine_similarities, 0)
    plt.imshow(cosine_similarities, cmap="plasma", interpolation="nearest")
    plt.colorbar()

    plt.title('Cosine Heatmap')
    plt.xlabel('Sentences')
    plt.ylabel('Sentences')

    plt.savefig("cosine_heatmap.png", format='png')
    plt.show()

    '''
    for record in records:
        gene_id = record["Gene Id"]
        summary = record["Summary"]

        
        summary_tokens = llama_tokenizer(
                    summary, 
                    return_tensors="pt",
                    add_special_tokens=False
                ).to(llama_model.device)

        summary_embeddings = llama_model.model.embed_tokens(summary_tokens["input_ids"])
    '''
