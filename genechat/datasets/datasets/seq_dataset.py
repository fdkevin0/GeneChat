import os
import sys
from genechat.datasets.datasets.base_dataset import BaseDataset
from torch.utils.data.dataloader import default_collate
import json
from torch.nn.utils.rnn import pad_sequence 
import torch
import random

questions = ["Tell me about this gene.", 
                "Please provide a detailed description of the gene."]
q_map = {
    "Which organism does the gene belong to?":
    " Limit your answer to one or two words.",
    "What is the locus type of the gene?":
    " Limit your answer to one or two words.",
    "On which chromosome is the gene located?":
    " Limit your answer to one or two words.",
    "How many exons does the gene contain?":
    " Limit your answer to one or two words.",
    "What is the official symbol of the gene?":
    " Limit your answer to one or two words.",
    "What is the official full name of the gene?":
    " Limit your answer to one or two words."
}
class SeqDataset(BaseDataset):
    def __init__(self, kw_path, text_rule_path, text_manual_path, seq_path):
        """
        protein (string): Root directory of protein (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        # print("______Enter Seq Dataset____")
        # super().__init__(vis_processor, text_processor)
        # self.qa_path = qa_path
        # self.seq_path = seq_path

        self.kw = json.load(open(kw_path, "r")) 
        self.rule = json.load(open(text_rule_path, "r"))
        self.manual = json.load(open(text_manual_path, "r"))
        self.sequence = json.load(open(seq_path, "r"))

        self.rate = {'kw':1, 'rule':1, 'manual':4}
        self.len_kw = len(self.kw)
        self.len_rule = len(self.rule)
        self.len_manual = len(self.manual)

        self.split1 = self.rate['kw'] * self.len_kw 
        self.split2 = self.split1 + self.rate['rule'] * self.len_rule
        self.split3 = self.split2 + self.rate['manual'] * self.len_manual 

    def __len__(self):
        return self.split3

    def __getitem__(self, index):
        
        if index < self.split1: # sample kw 
            gene_id = self.kw[index]["Gene Id"]
            answer = self.kw[index]["A"]
            query = self.kw[index]['Q']
            query += q_map[query]
            prompt = f"###Human: <gene>{gene_id}<geneHere></gene> {query} ###Assistant:"
        elif index < self.split2: # sample rule based functionality
            true_index  = (index - self.split1) % self.len_rule
            gene_id = self.rule[true_index]["Gene Id"]
            answer = self.rule[true_index]["Summary"]

            '''
            ########################################################################################################################################## - CHANGE 
            if 'Name' in self.rule[true_index]:
                name = self.rule[true_index]["Name"]
                prompt = f"###Human: <gene>{name}-{gene_id}<geneHere></gene> {random.choice(questions)} ###Assistant:"
            else:
                prompt = f"###Human: <gene>{gene_id}<geneHere></gene> {random.choice(questions)} ###Assistant:"
            ########################################################################################################################################## - CHANGE 
            '''
            prompt = f"###Human: <gene>:wq<geneHere></gene> {random.choice(questions)} ###Assistant:"
        else: # sample manual annotated functionality
            true_index  = (index - self.split2) % self.len_manual
            gene_id = self.manual[true_index]["Gene Id"]
            answer = self.manual[true_index]["Summary"]
        
        seq = self.sequence[gene_id]

        if len(seq[0]) > 160000:
            seq[0] = seq[0][:159999]

        return {
            "seq": seq,
            "text_input": answer,
            "prompt": prompt
        }

    # stage1-Qformer
        # gene_id = self.annotation[index]["gene_id"]
        # seq = self.sequence[gene_id]
        # answer = self.annotation[index]["name"]

        # if len(seq) > 1024:
        #     seq = seq[:1024]

        # return {
        #     "seq": seq,
        #     "text_input": answer
        # }