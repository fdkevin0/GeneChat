import logging
import esm

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
from argparse import ArgumentParser
import json

from genechat.common.registry import registry
from genechat.models.blip2 import Blip2Base, disabled_train
from genechat.models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer

#Import the Gene Encoder Libraries
from genechat.models.gene_encoder import HyenaDNAPreTrainedModel, CharacterTokenizer

#Transformer Modules
from transformers import AutoTokenizer, EsmModel, AutoModel
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

import time
from typing import List

@registry.register_model("genechat")
class GeneChat(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "",
    }

    def __init__(
        self,
        freeze_gene_encoder=True,
        gene_model="",
        max_gene_length=160000,
        gene_pool_width=1000,
        freeze_adaptor=False,
        freeze_llama=True,
        llama_model="",
        embedding_agg=1, 
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource
        self.embedding_agg = embedding_agg
        
        ######################################################################################################  <MAJOR CHANGE>
        '''
        ######################################################################################################  HYENADNA
        print('\n\n---->Loading Gene Encoder - HyenaDNA...')
        # we need these for the decoder head, if using
        use_head = False

        # you can override with your own backbone config here if you want,
        # otherwise we'll load the HF one in None
        backbone_cfg = None

        # Mode - Pool, Sum, Last
        mode = 'pool'

        # Max Length
        self.max_gene_length = max_gene_length

        # Gene Encoder - Hyena DNA Model
        self.gene_encoder = HyenaDNAPreTrainedModel.from_pretrained(
            './checkpoints',
            gene_model,
            download=True,
            config=backbone_cfg,
            device=torch.cuda.current_device(),
            use_head=use_head,
            mode=mode
        )

        # Gene Tokenizer
        self.gene_tokenizer = CharacterTokenizer(
            characters=['A', 'C', 'G', 'T', 'N'],  
            model_max_length=self.max_gene_length + 2,  
            padding_side='left', # since HyenaDNA is causal, we pad on the left
        )

        # Pooling layer to pool the output of HyenaDNA
        self.gene_pool_width = gene_pool_width
        self.avg_pool = torch.nn.AvgPool1d(kernel_size=gene_pool_width, stride=gene_pool_width)
        '''
    
        ######################################################################################################  DNABERT2
        print('\n\n---->Loading Gene Encoder - DNABERT2...')
        self.max_gene_length = max_gene_length
        self.gene_encoder = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        self.gene_encoder = self.gene_encoder.to(torch.cuda.current_device())

        self.gene_tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

        ######################################################################################################  </MAJOR CHANGE>
    
        if freeze_gene_encoder:
            #Freezing gene encoder parameters
            for name, param in self.gene_encoder.named_parameters():
                param.requires_grad = False
            self.gene_encoder = self.gene_encoder.eval()
            self.gene_encoder.train = disabled_train
            logging.info("freeze gene encoder")
        else:
            self.gene_encoder = self.gene_encoder.train()
        
        print('\n\n---->Loading LLAMA')

        # LLama Tokenizer
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        
        # LLama Model
        if self.low_resource:
            print("Start Low Resource Mode")
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map='auto'
                # device_map={'': device_8bit}
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model,
                torch_dtype=torch.float16,
            )

        if freeze_llama:
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
        else:
            lora_target_modules: List[str] = ["q_proj", "v_proj"]
            config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=lora_target_modules,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llama_model = get_peft_model(self.llama_model, config).model
            self.llama_model.print_trainable_parameters()

        # Linear layer to align the gene embeddings to the LLama token embedding space
        
        ######################################################################################################  </MAJOR CHANGE>
        ######################################################################################################  DNABERT2
        self.hyena_llama_proj = nn.Linear(
            self.gene_encoder.embeddings.word_embeddings.weight.shape[1], self.llama_model.config.hidden_size
        )

        '''
        ######################################################################################################  HyenaDNA
        self.hyena_llama_proj = nn.Linear(
            self.gene_encoder.backbone.embeddings.word_embeddings.embedding_dim, self.llama_model.config.hidden_size
        )
        '''
        ######################################################################################################  </MAJOR CHANGE>

        if freeze_adaptor:
            for name, param in self.hyena_llama_proj.named_parameters():
                param.requires_grad = False
        
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

    def encode_gene(self, seqs):
        '''
        Encode the input gene sequence using the HyenaDNA
        Parameters:
            seqs            - Batch of gene sequences
        Output:
            inputs_llama    - Encoded gene embedding which is projected to the LLama Embedding space
            atts_llam       - Attention Masks of the input tokens
        '''

        '''
        ######################################################################################################  HyenaDNA
        batch_seqs = []
        for seq in seqs:
            batch_seqs.append(seq)
        
        batch_tokenizer_output = self.gene_tokenizer(
                        batch_seqs,
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_gene_length,
                        return_tensors="pt"
                    )
        
        batch_tokens = batch_tokenizer_output["input_ids"]
        batch_tokens = batch_tokens.to(torch.cuda.current_device())

        # Extract the gene embeddings
        gene_embeds = self.gene_encoder(batch_tokens).to(batch_tokens.device)#self.gene_encoder(batch_tokens, repr_layers=[33], return_contacts=True)["representations"][33].to(batch_tokens.device)

        #Pooling the gene embeddings 
        #Output of the gene encoder is 160K. Pooling them in fixed intervals to get a smaller sequence of embeddings
        gene_embeddings_permute = gene_embeds.permute(0, 2, 1)
        gene_embeddings_permute = self.avg_pool(gene_embeddings_permute)
        gene_embeds = gene_embeddings_permute.permute(0, 2, 1)

        # input llama is of shape [B, len, 5120]
        if gene_embeds.dtype != self.hyena_llama_proj.weight.dtype:
            gene_embeds = gene_embeds.to(self.hyena_llama_proj.weight.dtype)

        #Alignment of gene embeddings to the LLama token embedding space
        inputs_llama = self.hyena_llama_proj(gene_embeds.squeeze(dim=2)).to(gene_embeds.device)

        # atts_llama is of shape [B, len]
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(gene_embeds.device)
        
        #print(f'Size of inputs_llama: {inputs_llama.size()}')
        #print(f'Size of atts_llama: {atts_llama.size()}')

        return inputs_llama, atts_llama

        '''
        
        ######################################################################################################  DNABERT2
        '''
        Encode the input gene sequence using the DNABERT2
        Parameters:
            seqs            - Batch of gene sequences
        Output:
            inputs_llama    - Encoded gene embedding which is projected to the LLama Embedding space
            atts_llam       - Attention Masks of the input tokens
        '''

        batch_seqs = []
        gene_embeds = []
        
        for seq in seqs:
            batch_seqs.append(seq)
            input_tokens = []
            for i in range(0, len(seq), 512):
                #Tokenize
                # (1,x)
                input_token = self.gene_tokenizer(seq[max(0, min(i, i-10)):i+512], return_tensors = 'pt')["input_ids"]
                input_token = input_token.to(self.gene_encoder.device)

                hidden_states = self.gene_encoder(input_token)[0]
                embedding_mean = torch.mean(hidden_states, dim=1)

                gene_embeds.append(embedding_mean)

        gene_embeds = torch.stack(gene_embeds, axis=1)
        
        # input llama is of shape [B, len, 5120]
        if gene_embeds.dtype != self.hyena_llama_proj.weight.dtype:
            gene_embeds = gene_embeds.to(self.hyena_llama_proj.weight.dtype)

        #Alignment of gene embeddings to the LLama token embedding space
        inputs_llama = self.hyena_llama_proj(gene_embeds.squeeze(dim=2)).to(gene_embeds.device)

        # atts_llama is of shape [B, len]
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(gene_embeds.device)
        
        return inputs_llama, atts_llama
        

    def prompt_list_wrap(self, img_embeds, atts_img, prompt):
        '''
        Wrap the gene embeddings with the pre-gene sequence and post-gene sequence
        Input to the LLama - pre-gene + gene-sequence + post-gene
        pre-gene:   'Given the gene sequence'
        post-gene:  'Question'
        
        Parameters:
            gene_embeds         -   aligned encoded gene embeddings
            atts_img            -   attention map of the gene tokens
            prompt              -   entire prompt: pre-gene + <geneHere> + post-gene
        Output:
            wrapped_gene_embeds -   embeddings of the wrapped gene sequence - llama input
            wrapped_atts_img    -   attention maps of the wrapped gene sequence - llama input
        '''

        if prompt:

            p_before_lst = []
            p_after_lst = []

            for p in prompt:
                p_before, p_after = p.split('<geneHere>')
                p_before_lst.append(p_before)
                p_after_lst.append(p_after)

            p_before_tokens_lst = self.llama_tokenizer(
                p_before_lst, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)

            p_after_tokens_lst = self.llama_tokenizer(
                p_after_lst, return_tensors="pt", add_special_tokens=True, padding=True).to(img_embeds.device)
            
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens_lst.input_ids)
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens_lst.input_ids)

            #print("\n", p_before_embeds.shape, img_embeds.shape, p_after_embeds.shape, "\n")
            #print("Prompt: ", self.llama_tokenizer.decode(torch.cat([p_before_tokens_lst.input_ids, p_after_tokens_lst.input_ids], dim=1)[0], add_special_tokens=False))

            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
            #wrapped_img_embeds = torch.cat([p_before_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img

    def forward(self, samples):
        seqs = samples["seq"][0] # list of seq
        gene_embeds, atts = self.encode_gene(seqs)

        #print("\n", samples, "\n", samples["text_input"], "\n")
        
        #gene_embeds, atts = torch.rand((1,1,5120), dtype=torch.float64).to(torch.cuda.current_device()), torch.ones((1,1), dtype=torch.long).to(torch.cuda.current_device())

        img_embeds, atts_img = self.prompt_list_wrap(gene_embeds, atts, samples["prompt"])

        self.llama_tokenizer.padding_side = "right"

        text = [t + self.end_sym for t in samples["text_input"]]
        
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(gene_embeds.device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                       dtype=torch.long).to(gene_embeds.device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

        #print("Input Embeds Shape: ", inputs_embeds.shape, " Gene Embeds shape: ", gene_embeds.shape, "Summary part shape: ", to_regress_embeds.shape, " Targets: ", targets.shape, "Attention Mask shape: ", attention_mask.shape)
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        logits = outputs.logits
        
        # print(torch.argmax(logits, dim=2).shape)
        logits = torch.argmax(logits, dim=2)
        loss = outputs.loss

        #print("Loss: ", loss.item())
        
        #print("===========")
        #print("Output: ", self.llama_tokenizer.batch_decode(logits, skip_special_tokens=True), "\n")
        #print("===========")
        #print("Input: ", self.llama_tokenizer.batch_decode(to_regress_tokens.input_ids[0,:], skip_special_tokens=True))
       
        ''' 
        with torch.no_grad():
            outputs = self.llama_model.generate(
                inputs_embeds= torch.cat([bos_embeds, img_embeds, to_regress_embeds[:,:5,:]], dim=1),
                max_new_tokens=128,
                num_beams=1,
                do_sample=False,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.9,
                length_penalty=1,
                temperature=float(0),
                output_hidden_states=False
            )
            output_token = outputs[0]

            print("Answer: ", self.llama_tokenizer.decode(to_regress_tokens.input_ids[0,:], add_special_tokens=False))
            print("Output Text: ", self.llama_tokenizer.decode(output_token, add_special_tokens=False))
            print("===========")        
        '''
        return {"loss": loss}

    @classmethod
    def from_config(cls, cfg):
        '''
        Get the configuration parameters from the config file
        '''
        
        llama_model = cfg.get("llama_model")

        gene_model=cfg.get("gene_model")
        freeze_gene_encoder = cfg.get("freeze_gene_encoder", False)
        max_gene_length=cfg.get("max_gene_length")
        gene_pool_width=cfg.get("gene_pool_width")

        freeze_adaptor = cfg.get("freeze_adaptor", False)

        freeze_llama = cfg.get("freeze_llama", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')
        embedding_agg = cfg.get("embedding_agg", 1)

        model = cls(
            freeze_gene_encoder=freeze_gene_encoder,
            gene_model=gene_model,
            max_gene_length=max_gene_length,
            gene_pool_width=gene_pool_width,
            freeze_adaptor=freeze_adaptor,
            freeze_llama=freeze_llama,
            llama_model=llama_model,
            embedding_agg = embedding_agg, 
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
        )
        
        stage1_ckpt = cfg.get("stage1_ckpt", "")  # load weights of encoder and adaptor layer
        if stage1_ckpt:
            print("\n\n------>Load HyenaDNA and adaptor layer Checkpoint: {}".format(stage1_ckpt))
            ckpt = torch.load(stage1_ckpt, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
            for key, value in ckpt['model'].items():
                if 'gene_encoder' in key:
                    print(key, value)
            #print(msg)
        peft_ckpt = cfg.get("peft_ckpt", "")  # load weights of LoRA
        if peft_ckpt:
            print("\n\n-------> Load LoRA Checkpoint: {}".format(peft_ckpt))
            ckpt = torch.load(peft_ckpt, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
            #print(msg)
        
        return model