import logging

#import sys
#sys.path.append('/nfs/mingjia/proteinchat_seq/anti_1b_code')

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
from argparse import ArgumentParser
import json

from genechat.common.registry import registry
from genechat.models.blip2 import Blip2Base, disabled_train
from genechat.models.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer

from anti_1b_code.estimated_ppl import get_embedding, initialize_model_and_tokenizer

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
        # vit_model="eva_clip_g",
        # q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        # img_size=224,
        # drop_path_rate=0,
        # use_grad_checkpoint=False,
        # vit_precision="fp16",
        freeze_gene_encoder=True,
        gene_model="",
        max_gene_length=160000,
        gene_pool_width=1000,
        freeze_adaptor = False,
        freeze_llama = True,
        # freeze_qformer=True,
        # num_query_token=32,
        llama_model="",
        embedding_agg=1, 
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource
        self.embedding_agg = embedding_agg
        
        print('\n\n---->Loading Gene Encoder...')

        parser = ArgumentParser()
        self.args_ = parser.parse_args()
        
        with open('train_configs/glm_config.txt', 'r') as f:    
            self.args_.__dict__ = json.load(f)
        
        self.args_.device = torch.cuda.current_device()

        if freeze_gene_encoder:
            self.args_.fp16 = True

        # Gene Encoder - Hyena DNA Model, Gene Tokenizer
        self.gene_encoder, self.gene_tokenizer = initialize_model_and_tokenizer(self.args_, freeze_gene_encoder)
   
        # Pooling layer to pool the output of HyenaDNA
        self.gene_pool_width = gene_pool_width
        self.avg_pool = torch.nn.AvgPool1d(kernel_size=gene_pool_width, stride=gene_pool_width)

        if freeze_gene_encoder:
            for name, param in self.gene_encoder.named_parameters():
                param.requires_grad = False

            self.gene_encoder = self.gene_encoder.eval()
            self.gene_encoder.train = disabled_train
            logging.info("freeze protein encoder")
        # else:
            # Debug, check model precision
            # print("GLM 130 B parameters")
            # for param in self.gene_encoder.parameters():
            #     print(param.dtype)
        #
        # print('Loading Q-Former')
        # self.Qformer, self.query_tokens = self.init_Qformer(
        #     num_query_token, self.visual_encoder.num_features
        # )
        # self.Qformer.cls = None
        # self.Qformer.bert.embeddings.word_embeddings = None
        # self.Qformer.bert.embeddings.position_embeddings = None
        # for layer in self.Qformer.bert.encoder.layer:
        #     layer.output = None
        #     layer.intermediate = None
        # self.load_from_pretrained(url_or_filename=q_former_model)
        #
        # if freeze_qformer:
        #     for name, param in self.Qformer.named_parameters():
        #         param.requires_grad = False
        #     self.Qformer = self.Qformer.eval()
        #     self.Qformer.train = disabled_train
        #     self.query_tokens.requires_grad = False
        #     logging.info("freeze Qformer")
        # print('Loading Q-Former Done')
        
        if "v1.5" in llama_model:
            print('\n\n---->Loading Vicuna based on LLaMA 2...')
        else:
            print('\n\n---->Loading Vicuna based on LLaMA 1...')

        # LLama Tokenizer
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)

        # LLama Model
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        
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
            
            self.llama_model = get_peft_model(self.llama_model, config)
            self.llama_model.print_trainable_parameters()

        # Linear layer to align the gene embeddings to the LLama token embedding space
        self.hyena_llama_proj = nn.Linear(
            self.gene_encoder.backbone.embeddings.word_embeddings.embedding_dim, self.llama_model.config.hidden_size
        )
        
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

        #Extract the gene embeddings
        gene_embeds = get_embedding(seqs, self.args_, self.embedding_agg, self.gene_encoder, self.gene_tokenizer)
        
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

    def prompt_list_wrap(self, gene_embeds, atts_img, prompt):
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
                p_before, p_after = p.split('<proteinHere>')
                p_before_lst.append(p_before)
                p_after_lst.append(p_after)

            p_before_tokens_lst = self.llama_tokenizer(p_before_lst, return_tensors="pt", add_special_tokens=False).to(gene_embeds.device)
            p_after_tokens_lst = self.llama_tokenizer(p_after_lst, return_tensors="pt", add_special_tokens=True, padding=True).to(gene_embeds.device)
            
            # lst = ['Describe this protein.', 'De', 'scribe', 'Des', 'cribe', 'Desc', 'ribe', 'Descr', 'ibe', 'Descri', 'be', 'Describ', 'e']
            # for i in range(len(lst)):
            #     ret = self.llama_tokenizer(
            #         lst[i], return_tensors="pt", add_special_tokens=True, padding=True).to(gene_embeds.device)
            #     print(lst[i])
            #     print(ret)
            # exit()
            # p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens_lst.input_ids)
            # p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens_lst.input_ids)
            
            p_before_embeds = self.llama_model.get_input_embeddings()(p_before_tokens_lst.input_ids)
            p_after_embeds = self.llama_model.get_input_embeddings()(p_after_tokens_lst.input_ids)

            wrapped_gene_embeds = torch.cat([p_before_embeds, gene_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_gene_embeds.shape[1])

            return wrapped_gene_embeds, wrapped_atts_img
        else:
            return gene_embeds, atts_img

    def forward(self, samples):
        '''
        Parameters:
            samples -   dict containing the prompt, sequence, answer
        Output:
            Loss    -   loss
        '''

        seqs = samples["seq"] # list of seq
        # print(samples)
        
        # Encode the genes
        gene_embeds, atts = self.encode_gene(seqs)

        #Wrap the genes with pre-gene and post-gene components
        prompt_embeds, atts_img = self.prompt_list_wrap(gene_embeds, atts, samples["prompt"])

        # LLama
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

        batch_size = prompt_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        
        # bos_embeds = self.llama_model.model.embed_tokens(bos)
        bos_embeds = self.llama_model.get_input_embeddings()(bos)

        atts_bos = atts_img[:, :1]

        # to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        to_regress_embeds = self.llama_model.get_input_embeddings()(to_regress_tokens.input_ids)

        inputs_embeds = torch.cat([bos_embeds, prompt_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

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
        #print(self.llama_tokenizer.batch_decode(logits, skip_special_tokens=True)[-400:])

        loss = outputs.loss
        return {"loss": loss}

    @classmethod
    def from_config(cls, cfg):
        '''
        Get the configuration parameters from the config file
        '''
        
        # vit_model = cfg.get("vit_model", "eva_clip_g")
        # q_former_model = cfg.get("q_former_model", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        # img_size = cfg.get("image_size")
        # num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        # drop_path_rate = cfg.get("drop_path_rate", 0)
        # use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        # vit_precision = cfg.get("vit_precision", "fp16")
        freeze_gene_encoder = cfg.get("freeze_gene_encoder", True)
        gene_model=cfg.get("gene_model"),
        max_gene_length=cfg.get("max_gene_length"),
        gene_pool_width=cfg.get("gene_pool_width"),

        freeze_adaptor = cfg.get("freeze_adaptor", False)

        freeze_llama = cfg.get("freeze_llama", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')
        embedding_agg = cfg.get("embedding_agg", 1)

        #print(embedding_agg)

        model = cls(
            # vit_model=vit_model,
            # q_former_model=q_former_model,
            # img_size=img_size,
            # drop_path_rate=drop_path_rate,
            # use_grad_checkpoint=use_grad_checkpoint,
            # vit_precision=vit_precision,
            freeze_gene_encoder=freeze_gene_encoder,
            gene_model=gene_model,
            max_gene_length=max_gene_length,
            gene_pool_width=gene_pool_width,
            freeze_llama=freeze_llama,
            freeze_adaptor=freeze_adaptor,
            # freeze_qformer=freeze_qformer,
            # num_query_token=num_query_token,
            llama_model=llama_model,
            embedding_agg = embedding_agg, 
            prompt_path=prompt_path,
            prompt_template=prompt_template,
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
        
        peft_ckpt = cfg.get("peft_ckpt", "")  # load weights of LoRA
        if peft_ckpt:
            print("\n\n-------> Load LoRA Checkpoint: {}".format(peft_ckpt))
            ckpt = torch.load(peft_ckpt, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)

        return model
