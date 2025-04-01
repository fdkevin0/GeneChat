import wandb
api = wandb.Api()

run1 = api.run("sdhana/GeneChat-DNABERT/nrjwoa2j")
run2 = api.run("sdhana/GeneChat-DNABERT/wf2bgleu")

history1 = run1.scan_history()
history2 = run2.scan_history()

losses1 = [row["loss"] for row in history1]
losses2 = [row["loss"] for row in history2]
losses = losses1 + losses2
n = 1

average_loss = [sum(losses[i:i+n]) / n for i in range(0, len(losses), n)]

wandb.init(project='GeneChat', name='Fine-tuning Vicuna-13B, Freeze DNABERT2 Average Loss')

for i in range(len(average_loss)):
    wandb.log({'step loss': average_loss[i]})

wandb.log({}, commit=True)
print(len(average_loss))
'''

import json

import torch
import numpy as np
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import torch.nn.functional as F


from transformers import AutoTokenizer, AutoModel

from transformers import AutoConfig, AutoModel, BertConfig

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#print(device)
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
model = model.to(torch.cuda.current_device())

dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
inputs = tokenizer(dna, padding="max_length", truncation=True, max_length=100, return_tensors = 'pt')["input_ids"]

inputs = inputs.to(torch.cuda.current_device())

hidden_states = model(inputs)[0] # [1, sequence_length, 768]

# embedding with mean pooling
embedding_mean = torch.mean(hidden_states[0], dim=0)
print(embedding_mean.shape) # expect to be 768

print(model)
print(model.embeddings.word_embeddings.weight.shape[1])

with open('/data2/gene_chat/exon_count/data/train_temp_set/seq.json', 'r') as file:
    data = json.load(file)

with torch.no_grad():
    batch_gene_embeds = []

    for key, value in data.items():
        #input = tokenizer(value[0], padding="max_length", truncation=True, max_length=160, return_tensors = 'pt')["input_ids"]
        #print(input.shape)
        #print("\n------\n",len(value[0]), len(value[0])/512)
        #gene_embeds = torch.empty((int(len(value[0]) / 1024) + 1, 768), dtype=torch.float32, device=model.device)
        #print(gene_embeds.shape)
        gene_embeds = []
        
        count = 0
        for i in range(0, len(value[0]), 512):
            input_token = tokenizer(value[0][max(0, min(i, i-10)):i+512], return_tensors = 'pt')["input_ids"]
            print(max(0, min(i, i-10)), i+512)

            input_token = input_token.to(model.device)
            
            hidden_states = model(input_token)[0]
            embedding_mean = torch.mean(hidden_states, dim=1)
    
            #gene_embeds[count] = embedding_mean[0]
            gene_embeds.append(embedding_mean)
            count += 1

        print(count)
        gene_embeds = torch.stack(gene_embeds, axis=1).squeeze(axis=0)
        print(gene_embeds.shape)

        batch_gene_embeds.append(gene_embeds)

    max_l = max(emb.shape[0] for emb in batch_gene_embeds)

    padded_batch_gene_embeds = [
    F.pad(emb, (0, 0, 0, max_l - emb.shape[0]), "constant", 0) for emb in batch_gene_embeds]

    batch_gene_embeds = torch.stack(padded_batch_gene_embeds)
    print(batch_gene_embeds.shape)

    #batch_gene_embeds = torch.stack(batch_gene_embeds)
 


 # for plot 

avg_emd=torch.mean(batch_gene_embeds,dim=1).cpu().numpy()

decom=TruncatedSVD(n_components=2)
decom_pl=decom.fit_transform(avg_emd)

plt.figure(figsize=(8, 6))
for i, embed in enumerate(decom_pl):
    plt.scatter(embed[0], embed[1], label=f"Sequence {i+1}", s=100)  
    plt.text(embed[0] + 0.02, embed[1] + 0.02, f"Seq {i+1}", fontsize=9) 

plt.title("Sequence Embedding")
plt.xlabel("axis 1")
plt.ylabel("axis 2")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
#plt.show()
plt.savefig("sequence_embedding.png", dpi=300)
'''