# Teaching Large Language Models Number-Focused Headline Generation With Key Element Rationales
The paper is published in NAACL 2025 Findings. A preprint is available on ArXiv [TEN-Sum](https://arxiv.org/abs/2502.03129).

## Method
![key steps](figures/key_steps.png)
Key steps for automatic generation of rationales to enhance numerical headline generation.

## Results
![result 1](figures/results_1.png)
![result 2](figures/results_2.png)
Numerical accuracy (%) and textual quality score (%) for TEN against baselines on NumHG and Xsum.

## Case Study
![case study](figures/case_study.png)
TEN vs. NCL (Baseline) for rationale and headline generation

## How to use

### 1 Install dependencies

    pip install -r requirements.txt

### 2 Dataset

The NumHG and Xsum datasetes are saved in the data folder. They have been augmented with the teacher LLM generated rationale.

### 3 Run SFT

#### Stage 1: SFT for rationale generator

Run the following code in terminal to start training

    accelerate launch --config_file recipes/multi_gpu.yaml --num_processes=1 scripts/run_sft.py recipes/config_sft_stg1.yaml
    
*note: if you have multiple gpus (e.g. 8), please change the flag --num_processes=8. Please also reduce the gradient_accumulation_steps parameter if you scale up the number of gpus.*

#### Stage 2: SFT for headline generator

Run the following code in terminal to start training

    accelerate launch --config_file recipes/multi_gpu.yaml --num_processes=1 scripts/run_sft.py recipes/config_sft_stg2.yaml
    
*note: if you have multiple gpus (e.g. 8), please change the flag --num_processes=8. Please also reduce the gradient_accumulation_steps parameter if you scale up the number of gpus.*


### 4 Run DPO
   
The preference data for dpo is saved in the data folder in our submission. We've created this dataset in our experiment. The process with a chartflow is detailed in our paper.

a. Merge the rationale generator's adapter to base model. Need to open and edit the file scripts/run_merge.py, and make the following changes:

    adapter_path = "results/models/numhg/ecnc/mistral/stg1-sft-r128a64lr2e4ep3x/checkpoint-{}" 
    output_path = "results/models/numhg/ecnc/mistral/stg1-sft-r128a64lr2e4ep3x-checkpoint-{}-merged"

*note: replace {} with the actual checkpoint number (i.e., the third and final checkpoint). It is from the model you trained in previous step 3 - stage 1.*

b. run the following code in terminal to merge the adapter

    python scripts/run_merge.py

c. to run dpo, taking mistral as example, need to open and edit the file recipes/config_dpo.yaml, and make change the model id

    model_id: results/models/numhg/ecnc/mistral/stg1-sft-r128a64lr2e4ep3x-checkpoint-{}-merged #fill in the merged model from step 4b.

d. run the following code in terminal to start the dpo process

    accelerate launch --config_file recipes/multi_gpu.yaml --num_processes=1 scripts/run_dpo.py recipes/config_dpo.yaml
    
*note: if you have multiple gpus (e.g. 8), please change the flag --num_processes=8. Please also reduce the gradient_accumulation_steps parameter if you scale up the number of gpus.*

### 5 Inference
   
#### Stage 1: generate the rationales for test data

a. open and edit the file recipes/config_inference_stg1.yaml, and make the following changes:
    
    model_path: results/models/numhg/ecnc/mistral/stg1-sft-r128a64lr2e4ep3x-checkpoint-{}-merged #merged model from step 4b.
    
    adapter_path: results/models/numhg/ecnc/mistral/stg1-dpo-r256a128lr2e6b08/checkpoint-{} 
    
*note: replace {} with the actual checkpoint number (i.e., the last checkpoint). It is from the model you trained in step 4d.*


b. run the following code in terminal to start the inference

    accelerate launch --config_file recipes/multi_gpu.yaml --num_processes=1 scripts/run_inference.py recipes/config_inference_stg1.yaml
    
*note: if you use multiple gpus (e.g. 8), please change the flag --num_processes=8.*
    
*note: if you use mulpiple gpus for inference, you will have to merge the output, by doing step c.*

c. if you have used multiple gpus for inference, run the following command

    python scripts/run_concat_stg1.py

#### Stage 2: generate the headlines

a. open and edit the file recipes/config_inference_stg2.yaml, and make the following changes:

    adapter_path: results/models/numhg/ecnc/mistral/stg2-sft-r64a32lr2e4ep3x/checkpoint-{} 
    
*note: replace {} with the actual checkpoint number (i.e., the third and final checkpoint). It is from the model you trained in step 3 - stage 2.*
    
    rationale_path: output/numhg/ecnc/mistral/rationales-stg1-dpo-r256a128lr2e6b08/rank-0 
    
*note: if you have used multipe gpus in step 5 - stage 1 - b and concated the results in step 5 - stage 1 - c, please make sure you used the concated file rank-c, i.e., rationale_path: output/numhg/ecnc/mistral/rationales-stg1-dpo-r256a128lr2e6b08/rank-c*

b. run the following code in terminal to start the inference

    accelerate launch --config_file recipes/multi_gpu.yaml --num_processes=1 scripts/run_inference.py recipes/config_inference_stg2.yaml
    
*note: if you have multiple gpus (e.g. 8), please change the flag --num_processes=8.*
*note: if you use mulpiple gpus for inference, you will have to merge the output, by doing step c.*

c. if you have used multiple gpus for inference, run the following command

    python scripts/run_concat_stg2.py


### Evaluation

We adopt the evaluation metrics commonly used in existing studies [(Huang et al., 2024)](http://arxiv.org/abs/2309.01455) to assess both the textual quality and numerical accuracy for headline generation. The metrics include number accuracy, ROUGE scores, BERTScores, and MoverScores. We adopt the [codes](https://github.com/ChunJiChen/NumEval_Evaluation) from [Huang et al. (2024)](http://arxiv.org/abs/2309.01455) to automatically calulate these metrics. 









