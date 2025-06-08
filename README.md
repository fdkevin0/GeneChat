# GeneChat: Multi-Modal Large Language Model Enables Gene Function Prediction

This repository contains the code and data of GeneChat: Multi-Modal Large Language Model Enables Gene Function Prediction [Manuscript](https://www.biorxiv.org/content/10.1101/2025.06.05.658031v1).

<!--
## Examples

 ![Eg1](fig/example.png)  

Examples of multi-round dialogues with ProteinChat for Q9U281, Q9XZG9, and Q9LU44.
-->

## Introduction
- GeneChat is a multi-modal large language model designed to predict gene descriptions from genomic sequences.
- GeneChat works in a similar way as ChatGPT. It takes as input the genomic sequence and predicts a description about the gene that includes which organism it might belong to, where it might be located and what it's functions are.
- The GeneChat model consists of a gene encoder, a large language model (LLM), and an adaptor. The gene encoder takes a genomic sequence as input and learns a representation for this gene. The adaptor transforms the gene representation produced by the gene encoder into LLM embedding space. The LLM takes the representation transformed by the adaptor and users' questions about this gene as inputs and generates answers. All these components are trained end-to-end. We use [DNABERT2](https://github.com/facebookresearch/esm) as the gene encoder.
- To train GeneChat, we designed (gene, prompt, answer) triplets from the NCBI dataset, resulting in ~51K genes.

![overview](fig/GeneChat.png)


## Getting Started
### Installation

**1. Prepare the code and the environment**

Git clone our repository, creating a python environment and ativate it via the following command

```bash
git clone https://github.com/Shashi-Sekar/GeneChat.git
cd GeneChat
conda env create -f environment.yml
conda activate genechat
```

Verify the installation of `torch` and `torchvision` is successful by running `python -c "import torchvision; print(torchvision.__version__)"`. If it outputs the version number without any warnings or errors, then you are good to go. __If it outputs any warnings or errors__, try to uninstall `torch` by `conda uninstall pytorch torchvision torchaudio cudatoolkit` and then reinstall them following [here](https://pytorch.org/get-started/previous-versions/#v1121). You need to find the correct command according to the CUDA version your GPU driver supports (check `nvidia-smi`). 

**2. Dataset**

The dataset contains 51,411 genes. It is curated from [NCBI](https://www.ncbi.nlm.nih.gov/gene). 
The collected data can be found on the drive [here](https://drive.google.com/drive/folders/1g0Pe0HxfzdhXWbG54rkd-Iya7c6wYZdO?usp=sharing)
You will see a `data` folder with two subfolders `train_set`, `test_set`.

**3. Prepare the pretrained Vicuna weights**

The current version of ProteinChat is built on Vicuna-13B-v1.5.
Please download Vicuna weights from [https://huggingface.co/lmsys/vicuna-13b-v1.5](https://huggingface.co/lmsys/vicuna-13b-v1.5).
Then, set the path to the vicuna weight in the config file
[configs/genechat_stage1.yaml](configs/genechat_stage1.yaml#L15).


### Training
**You need at least 70 GB GPU memory for the training.** 

The training configuration file is [configs/genechat_stage1.yaml](configs/genechat_stage1.yaml). In addition, you may want to change the number of epochs and other hyper-parameters there, such as `max_epoch`, `init_lr`, `min_lr`,`warmup_steps`, `batch_size_train`. Please adjust `iters_per_epoch` so that `iters_per_epoch` * `batch_size_train` = your training set size. 

Also, set your desired output directory [here](configs/proteinchat_stage1.yaml#53).

Start the training by running 
```bash
bash finetune.sh --cfg-path configs/genechat_stage1.yaml
``` 

### Evaluation

Modify the checkpoint paths in [configs/genechat_eval.yaml](configs/genechat_eval.yaml) to the location of your checkpoint.
We provide a stage1_ckpt [here](https://drive.google.com/drive/folders/1AaSzc9nlh_kfOJDuhLBfDHo3pGKrcKAE?usp=sharing) by training on 47,275 genes. peft_ckpt can be set empty during evaluation.

You can evaluate the model by running
```bash
bash demo.sh
``` 


## Acknowledgement

+ [DNABERT2](https://github.com/MAGICS-LAB/DNABERT_2)
+ [HyenaDNA](https://github.com/HazyResearch/hyena-dna)
+ [MiniGPT-4](https://minigpt-4.github.io/) 
+ [Lavis](https://github.com/salesforce/LAVIS)
+ [Vicuna](https://github.com/lm-sys/FastChat)


## License
This repository is under [BSD 3-Clause License](LICENSE.md).
