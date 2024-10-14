import json
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import AutoPeftModelForCausalLM, PeftModel, LoraConfig
import argparse
import yaml
import torch
import json
import random
import re
from compare_mt.rouge.rouge_scorer import RougeScorer
from nltk import sent_tokenize, word_tokenize
import time
from itertools import chain
from accelerate import PartialState, Accelerator
import os
from multiprocessing import cpu_count
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig, setup_chat_format
import wandb


def load_raw_data(data_path):

    with open(data_path) as f:
        data = json.load(f)

    return data

def create_dataset_for_training(data_q, task):

    def construct_messages(q, task):
        if task == "hg":
            msgs = [
                {"role": "system", "content": """You will be given a piece of news article after prefix 'news:'. Given the news article, please write an appropriate headline. Please ensure there is a digital number in the headline."""}, 
                {"role": "user", "content": None},
                {"role": "assistant", "content": None}
                ]
            msgs[1]["content"] = "news: " + str(q["news"]).strip()
            msgs[2]["content"] = str(q["headline"]).strip()

        if task == "rationale_hg_stg1_cal":
            msgs = [
                {"role": "system", "content": """You will be given a piece of news article after prefix 'news:'. Based on the news content, describe the calculation steps after the prefix 'calculation:'."""}, 
                {"role": "user", "content": None},
                {"role": "assistant", "content": None}
                ]
            msgs[1]["content"] = "news: " + str(q["news"]).strip()
            msgs[2]["content"] = "calculation: " + str(q["rationale_cal"]).strip()

        if task == "rationale_hg_stg1_emp":
            msgs = [
                {"role": "system", "content": """You will be given a piece of news article after prefix 'news:'. Based on the news content, please point out the key information a headline should focus on after prefix 'focus:'."""}, 
                {"role": "user", "content": None},
                {"role": "assistant", "content": None}
                ]
            msgs[1]["content"] = "news: " + str(q["news"]).strip()
            msgs[2]["content"] = "focus: " + str(q["rationale_hg"]).strip()           

        if task == "rationale_hg_stg1_ec":
            msgs = [
                {"role": "system", "content": """You will be given a piece of news article after prefix 'news:'. Based on the news content, please provide a brief summary after prefix 'summary:', point out the key information a headline should focus on after prefix 'focus:', and describe the calculation steps after the prefix 'calculation:'."""}, 
                {"role": "user", "content": None},
                {"role": "assistant", "content": None}
                ]
            msgs[1]["content"] = "news: " + str(q["news"]).strip()
            msgs[2]["content"] = "summary: " + str(q["summary"]).strip() + " focus: " + str(q["rationale_hg"]).strip() + " calculation: " + str(q["rationale_cal"])

        if task == "rationale_hg_stg1_ecnc":
            msgs = [
                {"role": "system", "content": """You will be given a piece of news article after prefix 'news:'. Based on the news content, emphasize what information the headline should focus on, list all the characters mentioned in the news content, list all the numerals mentioned in the news content, and explain how the numeral in the headline is calculated based on the characters and numerals you list."""}, 
                {"role": "user", "content": None},
                {"role": "assistant", "content": None}
                ]
            msgs[1]["content"] = "news: " + str(q["news"]).strip()
            msgs[2]["content"] = str(q["ecnc"]) .strip()

        if task == "rationale_hg_stg2_cal":
            msgs = [
                {"role": "system", "content": """You will be given a piece of news article after prefix 'news:'. You will be provided with a description of the calculation steps that leads to the number in the headline after the prefix 'calculation:'. Based on these information, please generate a numerically accurate headline."""}, 
                {"role": "user", "content": None},
                {"role": "assistant", "content": None}
                ]
            msgs[1]["content"] = "news: " + str(q["news"]).strip() + " calculation: " + str(q["rationale_cal"]).strip()
            msgs[2]["content"] = str(q["headline"]).strip()

        if task == "rationale_hg_stg2_emp":
            msgs = [
                {"role": "system", "content": """You will be given a piece of news article after prefix 'news:'. You will be given the key information a headline should focus on after prefix 'focus:'. Based on these information, please generate a numerically accurate headline."""}, 
                {"role": "user", "content": None},
                {"role": "assistant", "content": None}
                ]
            msgs[1]["content"] = "news: " + str(q["news"]).strip() + " focus: " + str(q["rationale_hg"]).strip()
            msgs[2]["content"] = str(q["headline"]).strip()


        if task == "rationale_hg_stg2_ec":
            msgs = [
                {"role": "system", "content": """You will be given a piece of news article after prefix 'news:'. You will be provided with a brief summary for the news after prefix 'summary:'. You will be given the key information a headline should focus on after prefix 'focus:'. You will be provided with a description of the calculation steps that leads to the number in the headline after the prefix 'calculation:'. Based on these information, please generate a numerically accurate headline."""}, 
                {"role": "user", "content": None},
                {"role": "assistant", "content": None}
                ]
            msgs[1]["content"] = "news: " + str(q["news"]).strip() + " summary: " + str(q["summary"]).strip() + " focus: " + str(q["rationale_hg"]).strip() + " calculation: " + str(q["rationale_cal"])
            msgs[2]["content"] = str(q["headline"]).strip()

        if task == "rationale_hg_stg2_ecnc":
            msgs = [
                {"role": "system", "content": """You will be given a piece of news article after prefix 'news:'. You will be given a hint about what information the headline should focus on, a list of all the characters mentioned in the news content, a list of all the numerals mentioned in the news content, and an explanation on how the numeral in the headline is calculated based on the character and numeral list. Based all the given information, please generate a numerically accurate headline."""}, 
                {"role": "user", "content": None},
                {"role": "assistant", "content": None}
                ]
            msgs[1]["content"] = "news: " + str(q["news"]).strip() + "\n\n" + str(q["ecnc"])
            msgs[2]["content"] = str(q["headline"]).strip()

        if task == "nr":
            msgs = [
                {"role": "system", "content": """You will be given a piece of news article after prefix 'news:'. You will be given a masked headline after prefix 'masked headline'. Based on the news content, please output the missing number in the masked headline."""}, 
                {"role": "user", "content": None},
                {"role": "assistant", "content": None}
                ]
    
            msgs[1]["content"] = "news: " + str(q["news"]).strip() + " masked headline: " + str(q["masked headline"]).strip()
            msgs[2]["content"] = str(q["ans"]).strip()

        if task == "rationale_nr_stg1_ec":
            msgs = [
                {"role": "system", "content": """You will be given a piece of news article after prefix 'news:'. You will be given a masked headline after prefix 'masked headline'. Based on the news content, please provide a brief summary after prefix 'summary:', point out the key information a headline should focus on after prefix 'focus:', and describe the calculation steps leading to the missing number in the headline after the prefix 'calculation:'."""}, 
                {"role": "user", "content": None},
                {"role": "assistant", "content": None}
                ]
    
            msgs[1]["content"] = "news: " + str(q["news"]).strip() + " masked headline: " + str(q["masked headline"]).strip()
            msgs[2]["content"] = "summary: " + str(q["summary"]).strip() + " focus: " + str(q["rationale_hg"]).strip() + " calculation: " + str(q["rationale_cal"]).strip()

        if task == "rationale_nr_stg1_ecnc":
            msgs = [
                {"role": "system", "content": """You will be given a piece of news article after prefix 'news:'. You will be given a masked headline after prefix 'masked headline'. Based on the given news, please emphasize what information the headline should focus on, list all the characters mentioned in the news content, list all the numerals mentioned in the news content, and explain how the numeral in the headline is calculated based on the characters and numerals you list."""}, 
                {"role": "user", "content": None},
                {"role": "assistant", "content": None}
                ]
    
            msgs[1]["content"] = "news: " + str(q["news"]).strip() + " masked headline: " + str(q["masked headline"]).strip()
            msgs[2]["content"] = str(q["ecnc"]).strip()


        if task == "rationale_nr_stg2_ec":
            msgs = [
                {"role": "system", "content": """You will be given a piece of news article after prefix 'news:'. You will be given a masked headline after prefix 'masked headline'. You will be given a brief summary after prefix 'summary:'. You will be given the key information a headline should focus on after prefix 'focus:'. You will be given the calculation steps leading to the missing number in the headline after the prefix 'calculation:'. Based on the information, please calulate the missing numeral in the headline"""}, 
                {"role": "user", "content": None},
                {"role": "assistant", "content": None}
                ]
    
            msgs[1]["content"] = "news: " + str(q["news"]).strip() + " masked headline: " + str(q["masked headline"]).strip() + " summary: " + str(q["summary"]).strip() + " focus: " + str(q["rationale_hg"]).strip() + " calculation: " + str(q["rationale_cal"]).strip()
            msgs[2]["content"] = str(q["ans"]).strip()

        if task == "rationale_nr_stg2_ecnc":
            msgs = [
                {"role": "system", "content": """You will be given a piece of news article after prefix 'news:'. You will be given a masked headline after prefix 'masked headline'. You will be given a hint about what information the headline should focus on, a list of all the characters mentioned in the news content, a list of all the numerals mentioned in the news content, and an explanation on how the numeral in the headline is calculated based on the character and numeral list. Based all the given information, please calculate the missing numeral in the headline."""}, 
                {"role": "user", "content": None},
                {"role": "assistant", "content": None}
                ]
    
            msgs[1]["content"] = "news: " + str(q["news"]).strip() + "\n\n" + str(q["ecnc"]).strip()
            msgs[2]["content"] = str(q["ans"]).strip()


        return msgs

    messages = []

    for q in data_q:
        msgs = construct_messages(q, task)
        messages.append(msgs)

    new_dataset = Dataset.from_dict({"messages":messages})

    return new_dataset


def load_sft_dataset(data_path, task):
    """
    task: sft
    """

    raw_data = load_raw_data(data_path)
    sft_dataset = create_dataset_for_training(raw_data, task)

    return sft_dataset
    
def merge_adapter(adapter_path, output_path): 
    
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    model = AutoPeftModelForCausalLM.from_pretrained(adapter_path)

    model = model.merge_and_unload()
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print("adapter merged...")


def read_args_sft():

    def load_yaml(yaml_path):
        with open(yaml_path, 'r') as file:
            return yaml.safe_load(file)
        
    yaml_parser = argparse.ArgumentParser()
    yaml_parser.add_argument('recipes', type=load_yaml)
    yaml_args = yaml_parser.parse_args()

    data_parser = argparse.ArgumentParser()
    data_parser.add_argument('--train_path')
    data_parser.add_argument('--eval_path')
    data_parser.add_argument('--task')


    model_parser = argparse.ArgumentParser()
    model_parser.add_argument('--model_id')
    model_parser.add_argument('--tokenizer_model_max_length')
    model_parser.add_argument('--tokenizer_padding_side')
    model_parser.add_argument('--tokenizer_pad_token')

    sftconfig_parser = argparse.ArgumentParser()
    sftconfig_parser.add_argument('--max_seq_length')
    sftconfig_parser.add_argument('--packing')
    sftconfig_parser.add_argument('--max_grad_norm')
    sftconfig_parser.add_argument('--bf16')
    sftconfig_parser.add_argument('--tf32')
    sftconfig_parser.add_argument('--do_eval')
    sftconfig_parser.add_argument('--eval_strategy')
    sftconfig_parser.add_argument('--gradient_accumulation_steps')
    sftconfig_parser.add_argument('--gradient_checkpointing')
    sftconfig_parser.add_argument('--gradient_checkpointing_kwargs')
    sftconfig_parser.add_argument('--learning_rate')
    sftconfig_parser.add_argument('--log_level')
    sftconfig_parser.add_argument('--logging_steps')
    sftconfig_parser.add_argument('--logging_strategy')
    sftconfig_parser.add_argument('--lr_scheduler_type')
    sftconfig_parser.add_argument('--max_steps')
    sftconfig_parser.add_argument('--num_train_epochs')
    sftconfig_parser.add_argument('--output_dir')
    sftconfig_parser.add_argument('--overwrite_output_dir')
    sftconfig_parser.add_argument('--per_device_eval_batch_size')
    sftconfig_parser.add_argument('--per_device_train_batch_size')
    sftconfig_parser.add_argument('--save_strategy')
    sftconfig_parser.add_argument('--save_total_limit')
    sftconfig_parser.add_argument('--seed')
    sftconfig_parser.add_argument('--warmup_ratio')
    sftconfig_parser.add_argument('--load_best_model_at_end')
    sftconfig_parser.add_argument('--dataset_kwargs')
    sftconfig_parser.add_argument('--report_to')

    peft_parser = argparse.ArgumentParser()
    peft_parser.add_argument('--r')
    peft_parser.add_argument('--lora_alpha')
    peft_parser.add_argument('--lora_dropout')
    peft_parser.add_argument('--bias')
    peft_parser.add_argument('--task_type')
    peft_parser.add_argument('--target_modules')

    data_args, _ = data_parser.parse_known_args()
    model_args, _ = model_parser.parse_known_args()
    sftconfig_args, _ = sftconfig_parser.parse_known_args()
    peft_args, _ = peft_parser.parse_known_args()

    if yaml_args.recipes:
        for key, value in yaml_args.recipes.items():
            if hasattr(data_args, key):
                setattr(data_args, key, value)

            if hasattr(model_args, key):
                setattr(model_args, key, value)    

            if hasattr(sftconfig_args, key):
                setattr(sftconfig_args, key, value)

            if hasattr(peft_args, key):
                setattr(peft_args, key, value)

    return data_args, model_args, sftconfig_args, peft_args


def read_args_inference():

    def load_yaml(yaml_path):
        with open(yaml_path, 'r') as file:
            return yaml.safe_load(file)
        
    yaml_parser = argparse.ArgumentParser()
    yaml_parser.add_argument('recipes', type=load_yaml)
    yaml_args = yaml_parser.parse_args()

    inf_parser = argparse.ArgumentParser()
    inf_parser.add_argument('--task')    
    inf_parser.add_argument('--model_path')
    inf_parser.add_argument('--adapter_path')
    inf_parser.add_argument('--data_path')
    inf_parser.add_argument('--rationale_path')
    inf_parser.add_argument('--output_dir')        
    inf_parser.add_argument('--sample_num')
    inf_parser.add_argument('--num_check')
    inf_parser.add_argument('--max_new_tokens')
    inf_parser.add_argument('--temperature')
    inf_parser.add_argument('--batch_size')
    inf_parser.add_argument('--inference_num')
    inf_parser.add_argument('--duplicates_multiple')

    inf_args, _ = inf_parser.parse_known_args()

    if yaml_args.recipes:
        for key, value in yaml_args.recipes.items():
            if hasattr(inf_args, key):
                setattr(inf_args, key, value)

    return inf_args


def read_args_dpo():

    def load_yaml(yaml_path):
        with open(yaml_path, 'r') as file:
            return yaml.safe_load(file)
        
    yaml_parser = argparse.ArgumentParser()
    yaml_parser.add_argument('recipes', type=load_yaml)
    yaml_args = yaml_parser.parse_args()

    dpo_parser = argparse.ArgumentParser()
    dpo_parser.add_argument('--task')
    dpo_parser.add_argument('--data_path')
    dpo_parser.add_argument('--model_id')
    dpo_parser.add_argument('--r')
    dpo_parser.add_argument('--lora_alpha')
    dpo_parser.add_argument('--lora_dropout')
    dpo_parser.add_argument('--bias')
    dpo_parser.add_argument('--task_type')
    dpo_parser.add_argument('--target_modules')
    dpo_parser.add_argument('--max_prompt_length')
    dpo_parser.add_argument('--max_length')
    dpo_parser.add_argument('--beta')
    dpo_parser.add_argument('--remove_unused_columns')
    dpo_parser.add_argument('--do_eval')
    dpo_parser.add_argument('--eval_strategy')
    dpo_parser.add_argument('--eval_steps')
    dpo_parser.add_argument('--per_device_eval_batch_size')
    dpo_parser.add_argument('--per_device_train_batch_size')
    dpo_parser.add_argument('--gradient_accumulation_steps')
    dpo_parser.add_argument('--gradient_checkpointing')
    dpo_parser.add_argument('--gradient_checkpointing_kwargs')
    dpo_parser.add_argument('--learning_rate')
    dpo_parser.add_argument('--max_grad_norm')
    dpo_parser.add_argument('--lr_scheduler_type')
    dpo_parser.add_argument('--max_steps')
    dpo_parser.add_argument('--num_train_epochs')
    dpo_parser.add_argument('--save_steps')
    dpo_parser.add_argument('--save_strategy')
    dpo_parser.add_argument('--save_total_limit')
    dpo_parser.add_argument('--log_level')
    dpo_parser.add_argument('--logging_steps')
    dpo_parser.add_argument('--output_dir')
    dpo_parser.add_argument('--optim')
    dpo_parser.add_argument('--warmup_ratio')
    dpo_parser.add_argument('--bf16')
    dpo_parser.add_argument('--seed')
    dpo_parser.add_argument('--load_best_model_at_end')
    dpo_parser.add_argument('--report_to')

    dpo_args, _ = dpo_parser.parse_known_args()

    if yaml_args.recipes:
        for key, value in yaml_args.recipes.items():
            if hasattr(dpo_args, key):
                setattr(dpo_args, key, value)

    return dpo_args


def load_pipline(model_path, adapter_path = None):

    device_string = PartialState().process_index

    quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # load the model and tokenizer
    if adapter_path:
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, quantization_config=quantization_config, attn_implementation="flash_attention_2", device_map={'':device_string})
        base_model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
    else: 
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, quantization_config=quantization_config, attn_implementation="flash_attention_2", device_map={'':device_string})
        # model.resize_token_embeddings(len(tokenizer))
    
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    print("pipeline successfully loaded...")

    return pipe

def load_data(data_path):

    # load the test data
    with open(data_path) as f:
        data = json.load(f)

    print("test data loaded...")
    print("example: ")
    print("news: ", data[0]["news"])
    # print("headline: ", data[0]["headline"])
    print("\n\n")

    return data
    
def cal_rouge(hyp, ref):

    def process(x):
        return sent_tokenize(" ".join(word_tokenize(x.strip())))

    hyp = hyp.strip().strip("\"")
    ref = ref.strip().strip("\"")

    hyp = process(hyp)
    ref = process(ref)

    rouge_scorer = RougeScorer(['rouge1'], use_stemmer=True)
    score = rouge_scorer.score("\n".join(ref), "\n".join(hyp))

    return score["rouge1"].fmeasure


def create_dataset_for_inference(data_q, task):

    def construct_messages(q, task):
        if task == "hg":
            msgs = [
                {"role": "system", "content": """You will be given a piece of news article after prefix 'news:'. Given the news article, please write an appropriate headline. Please ensure there is a digital number in the headline."""}, 
                {"role": "user", "content": None},
                {"role": "assistant", "content": None}
                ]
            msgs[1]["content"] = "news: " + str(q["news"]).strip()
            msgs[2]["content"] = str(q["headline"]).strip()

        if task == "rationale_hg_stg1_cal":
            msgs = [
                {"role": "system", "content": """You will be given a piece of news article after prefix 'news:'. Based on the news content, please describe the calculation steps after the prefix 'calculation:'."""}, 
                {"role": "user", "content": None},
                {"role": "assistant", "content": None}
                ]
            msgs[1]["content"] = "news: " + str(q["news"]).strip()
            msgs[2]["content"] = "calculation: " + str(q["rationale_cal"])

        if task == "rationale_hg_stg1_emp":
            msgs = [
                {"role": "system", "content": """You will be given a piece of news article after prefix 'news:'. Based on the news content, please point out the key information a headline should focus on after prefix 'focus:'."""}, 
                {"role": "user", "content": None},
                {"role": "assistant", "content": None}
                ]
            msgs[1]["content"] = "news: " + str(q["news"]).strip()
            msgs[2]["content"] = "focus: " + str(q["rationale_hg"]).strip()


        if task == "rationale_hg_stg1_ec":
            msgs = [
                {"role": "system", "content": """You will be given a piece of news article after prefix 'news:'. Based on the news content, please provide a brief summary after prefix 'summary:', point out the key information a headline should focus on after prefix 'focus:', and describe the calculation steps after the prefix 'calculation:'."""}, 
                {"role": "user", "content": None},
                {"role": "assistant", "content": None}
                ]
            msgs[1]["content"] = "news: " + str(q["news"]).strip()
            msgs[2]["content"] = "summary: " + str(q["summary"]).strip() + " focus: " + str(q["rationale_hg"]).strip() + " calculation: " + str(q["rationale_cal"])


        if task == "rationale_hg_stg1_ecnc":
            msgs = [
                {"role": "system", "content": """You will be given a piece of news article after prefix 'news:'. Based on the news content, emphasize what information the headline should focus on, list all the characters mentioned in the news content, list all the numerals mentioned in the news content, and explain how the numeral in the headline is calculated based on the characters and numerals you list."""}, 
                {"role": "user", "content": None},
                {"role": "assistant", "content": None}
                ]
            msgs[1]["content"] = "news: " + str(q["news"]).strip()
            msgs[2]["content"] = str(q["ecnc"]) .strip()

        if task == "rationale_hg_stg2_cal":
            msgs = [
                {"role": "system", "content": """You will be given a piece of news article after prefix 'news:'. You will be provided with a description of the calculation steps that leads to the number in the headline after the prefix 'calculation:'. Based on these information, please generate a numerically accurate headline."""}, 
                {"role": "user", "content": None},
                {"role": "assistant", "content": None}
                ]
            msgs[1]["content"] = "news: " + str(q["news"]).strip() + " " + str(q["rationale_hg_stg1"]).strip()
            msgs[2]["content"] = str(q["headline"]).strip()

        if task == "rationale_hg_stg2_emp":
            msgs = [
                {"role": "system", "content": """You will be given a piece of news article after prefix 'news:'. You will be given the key information a headline should focus on after prefix 'focus:'. Based on these information, please generate a numerically accurate headline."""}, 
                {"role": "user", "content": None},
                {"role": "assistant", "content": None}
                ]
            msgs[1]["content"] = "news: " + str(q["news"]).strip() + " " + str(q["rationale_hg_stg1"]).strip()
            msgs[2]["content"] = str(q["headline"]).strip()


        if task == "rationale_hg_stg2_ec":
            msgs = [
                {"role": "system", "content": """You will be given a piece of news article after prefix 'news:'. You will be provided with a brief summary for the news after prefix 'summary:'. You will be given the key information a headline should focus on after prefix 'focus:'. You will be provided with a description of the calculation steps that leads to the number in the headline after the prefix 'calculation:'. Based on these information, please generate a numerically accurate headline."""}, 
                {"role": "user", "content": None},
                {"role": "assistant", "content": None}
                ]
            msgs[1]["content"] = "news: " + str(q["news"]).strip() + " " + str(q["rationale_hg_stg1"]).strip()
            msgs[2]["content"] = str(q["headline"]).strip()

        if task == "rationale_hg_stg2_ecnc":
            msgs = [
                {"role": "system", "content": """You will be given a piece of news article after prefix 'news:'. You will be given a hint about what information the headline should focus on, a list of all the characters mentioned in the news content, a list of all the numerals mentioned in the news content, and an explanation on how the numeral in the headline is calculated based on the character and numeral list. Based all the given information, please generate a numerically accurate headline."""}, 
                {"role": "user", "content": None},
                {"role": "assistant", "content": None}
                ]
            msgs[1]["content"] = "news: " + str(q["news"]).strip() + "\n\n" + str(q["rationale_hg_stg1"]).strip()
            msgs[2]["content"] = str(q["headline"]).strip()


        if task == "nr":
            msgs = [
                {"role": "system", "content": """You will be given a piece of news article after prefix 'news:'. You will be given a masked headline after prefix 'masked headline'. Based on the news content, please output the missing number in the masked headline."""}, 
                {"role": "user", "content": None},
                {"role": "assistant", "content": None}
                ]
    
            msgs[1]["content"] = "news: " + str(q["news"]).strip() + " masked headline: " + str(q["masked headline"]).strip()
            msgs[2]["content"] = str(q["ans"]).strip()

        if task == "rationale_nr_stg1_ec":
            msgs = [
                {"role": "system", "content": """You will be given a piece of news article after prefix 'news:'. You will be given a masked headline after prefix 'masked headline'. Based on the given news, Based on the news content, please provide a brief summary after prefix 'summary:', point out the key information a headline should focus on after prefix 'focus:', and describe the calculation steps leading to the missing number in the headline after the prefix 'calculation:'."""}, 
                {"role": "user", "content": None},
                {"role": "assistant", "content": None}
                ]
    
            msgs[1]["content"] = "news: " + str(q["news"]).strip() + " masked headline: " + str(q["masked headline"]).strip()
            msgs[2]["content"] = "summary: " + str(q["summary"]).strip() + " focus: " + str(q["rationale_hg"]).strip() + " calculation: " + str(q["rationale_cal"]).strip()

        if task == "rationale_nr_stg1_ecnc":
            msgs = [
                {"role": "system", "content": """You will be given a piece of news article after prefix 'news:'. You will be given a masked headline after prefix 'masked headline'. Based on the given news, please emphasize what information the headline should focus on, list all the characters mentioned in the news content, list all the numerals mentioned in the news content, and explain how the numeral in the headline is calculated based on the characters and numerals you list."""}, 
                {"role": "user", "content": None},
                {"role": "assistant", "content": None}
                ]
    
            msgs[1]["content"] = "news: " + str(q["news"]).strip() + " masked headline: " + str(q["masked headline"]).strip()
            msgs[2]["content"] = str(q["ecnc"]).strip()

        if task == "rationale_nr_stg2_ec":
            msgs = [
                {"role": "system", "content": """You will be given a piece of news article after prefix 'news:'. You will be given a masked headline after prefix 'masked headline'. You will be given a brief summary after prefix 'summary:'. You will be given the key information a headline should focus on after prefix 'focus:'. You will be given the calculation steps leading to the missing number in the headline after the prefix 'calculation:'."""}, 
                {"role": "user", "content": None},
                {"role": "assistant", "content": None}
                ]
    
            msgs[1]["content"] = "news: " + str(q["news"]).strip() + " masked headline: " + str(q["masked headline"]).strip() + " " + str(q["rationale_nr_stg1"]).strip()
            msgs[2]["content"] = str(q["ans"]).strip()


        if task == "rationale_nr_stg2_ecnc":
            msgs = [
                {"role": "system", "content": """You will be given a piece of news article after prefix 'news:'. You will be given a masked headline after prefix 'masked headline'. You will be given a hint about what information the headline should focus on, a list of all the characters mentioned in the news content, a list of all the numerals mentioned in the news content, and an explanation on how the numeral in the headline is calculated based on the character and numeral list. Based all the given information, please calculate the missing numeral in the headline."""}, 
                {"role": "user", "content": None},
                {"role": "assistant", "content": None}
                ]
    
            msgs[1]["content"] = "news: " + str(q["news"]).strip() + "\n\n" + str(q["rationale_nr_stg1"]).strip()
            msgs[2]["content"] = str(q["ans"]).strip()



        return msgs

    messages = []

    for q in data_q:
        msgs = construct_messages(q, task)
        messages.append(msgs)

    new_dataset = Dataset.from_dict({"messages":messages})

    accelerator = Accelerator()
    num_gpus = accelerator.num_processes
    total_samples = len(new_dataset)
    chunk_size = total_samples // num_gpus

    start_idx = accelerator.process_index * chunk_size
    end_idx = start_idx + chunk_size if accelerator.process_index != num_gpus - 1 else total_samples
    dataset_shard = new_dataset.select(range(start_idx, end_idx))

    return dataset_shard


def single_inference(pipe, sample, max_new_tokens, temperature, num_check):

    num_check_runs = 0
    
    prompt = pipe.tokenizer.apply_chat_template(sample["messages"][:2], tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, top_k=50, top_p=0.95, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)
    new_tokens = str(outputs[0]['generated_text'][len(prompt):]).strip()

    while (num_check == True) and (num_check_runs < 20):
        pattern = re.compile(r'\d{1,3}(?:,\d{3})+|\d+[\/\.]{0,1}\d+|\d+')
        generated_num_list = pattern.findall(new_tokens)
        if len(generated_num_list) == 1:
            num_check = False
        else:
            outputs = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, top_k=50, top_p=0.95, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id)
            new_tokens = str(outputs[0]['generated_text'][len(prompt):]).strip()
            num_check_runs += 1             

    return new_tokens


def test_inference(pipe, dataset, max_new_tokens, sample_num, temperature=0.01, num_check=False):
    
    for index in random.sample(range(len(dataset)), sample_num):     
        new_tokens = single_inference(pipe, dataset[index], max_new_tokens, temperature, num_check)
        print(f"News:\n{dataset[index]['messages'][1]['content']}\n")
        print(f"Groud truth:\n{dataset[index]['messages'][2]['content']}\n")
        print(f"Generated Answer:\n{new_tokens}\n")


def bulk_inference(pipe, dataset, max_new_tokens, temperature, batch_size, inference_num=1):

        new_tokens_list = []

        def promptIterator():
                for sample in dataset:
                        for i in range(inference_num):
                            prompt = pipe.tokenizer.apply_chat_template(sample["messages"][:2], tokenize=False, add_generation_prompt=True)
                            yield prompt

        prompt_length = [len(prompt) for prompt in promptIterator()]
        len_prompt_len = len(prompt_length)
        print(f"Total {len_prompt_len} infrerences\n")

        start_time = time.time()

        for outputs, length in zip(pipe(promptIterator(), max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, top_k=50, top_p=0.95, eos_token_id=pipe.tokenizer.eos_token_id, pad_token_id=pipe.tokenizer.pad_token_id, batch_size=batch_size), prompt_length):
                new_tokens = outputs[0]['generated_text'][length:].strip()
                new_tokens_list.append(new_tokens)
                len_new_tokens_list = len(new_tokens_list)
                
                if len_new_tokens_list == batch_size:
                        seconds = (time.time()-start_time)/batch_size*(len_prompt_len-batch_size)
                        minutes = round(seconds/60, 2)
                        hours = round(minutes/60, 2)
                        print(f"Estimated time: {minutes} minutes or {hours} hours\n")

                if len_new_tokens_list in [int(len_prompt_len*i/10) for i in range(1, 10)]:
                        completion = round(len_new_tokens_list/len_prompt_len*100, 2)
                        seconds = (time.time()-start_time)/len_new_tokens_list*(len_prompt_len-len_new_tokens_list)
                        minutes = round(seconds/60, 2)
                        hours = round(minutes/60, 2)
                        print(f"Completion: {completion}...\nRemaining time: {minutes} minutes or {hours} hours\n")

        return new_tokens_list

def save_inference(new_tokens_list, output_dir, task):

    os.makedirs(output_dir, exist_ok=True)

    if "stg1" in task:

        output_file = f"{output_dir}/rank-{Accelerator().process_index}.json"
        with open(output_file, "w") as j:
            json.dump(new_tokens_list, j)

    else:
        output_file = f"{output_dir}/rank-{Accelerator().process_index}.txt"
        with open(output_file, "w") as t:
            new_tokens_list = [line.replace("\n", " ") for line in new_tokens_list]
            t.write("\n".join(new_tokens_list))       
            
    print("inferences saved...")


def concat_output_files(output_dir, task, num_processes):

    if "stg1" in task:
        file_type = "json"
    else:
        file_type = "txt"

    with open(f"{output_dir}/rank-c.{file_type}", "w") as cat:
        
        combined = []

        for i in range(num_processes):
            with open(f"{output_dir}/rank-{i}.{file_type}") as file:
                if file_type == "json":
                    inferences = json.load(file)
                else:
                    inferences = [line.strip() for line in file]
            combined =  combined + inferences

        if file_type == "json":
            json.dump(combined, cat)
        else:
            cat.write("\n".join(combined))


def create_dataset_stg2(dataset, rationale_path, task, duplicates_multiple):

    with open(rationale_path) as f:
        stg1 = json.load(f)

    if duplicates_multiple:
        dataset = [[sample]*duplicates_multiple for sample in dataset]
        dataset = list(chain.from_iterable(dataset))

    if "hg" in task:
        dataset_stg2 = [dict(sample, rationale_hg_stg1=generated_r) for sample, generated_r in zip(dataset, stg1)]
    elif "nr" in task:
        dataset_stg2 = [dict(sample, rationale_nr_stg1=generated_r) for sample, generated_r in zip(dataset, stg1)]

    return dataset_stg2


def prepare_dataset_inference(data_path, rationale_path, task, duplicates_multiple):

    dataset = load_data(data_path)

    if "stg2" in task:

        dataset_stg2 = create_dataset_stg2(dataset, rationale_path, task, duplicates_multiple)
        dataset_inference = create_dataset_for_inference(dataset_stg2, task)

    else:
        dataset_inference = create_dataset_for_inference(dataset, task)

    for role in dataset_inference[0]["messages"]:
        print(role["content"], "\n")

    
    return dataset_inference


def apply_dpo_chat_template(data_dpo, tokenizer, task):

    system_message = {
        "rationale_hg_stg1_ecnc":{"role": "system", "content": """You will be given a piece of news article after prefix 'news:'. Based on the news content, emphasize what information the headline should focus on, list all the characters mentioned in the news content, list all the numerals mentioned in the news content, and explain how the numeral in the headline is calculated based on the characters and numerals you list."""},
        "hg": {"role": "system", "content": """You will be given a piece of news article after prefix 'news:'. Given the news article, please write an appropriate headline. Please ensure there is a digital number in the headline."""},
    }


    def apply_chat_template(example, tokenizer):

        #format system
        message = system_message[task]
        system = tokenizer.apply_chat_template([message], tokenize=False)

        #format instruction
        message = {"role": "user", "content": "news: " + str(example['news'])}
        prompt = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)

        #format chosen answer
        chosen = str(example["chosen"]) + f"{tokenizer.eos_token}\n"

        #format rejected answer
        rejected = str(example["rejected"]) + f"{tokenizer.eos_token}\n"

        example["prompt"] = system + prompt
        example["chosen"] = chosen
        example["rejected"] = rejected

        return example

    data_dpo_train = data_dpo["train"].map(apply_chat_template,
                                    num_proc=cpu_count(),
                                    fn_kwargs={"tokenizer": tokenizer},
                                    remove_columns=['news'],
                                    desc="Applying chat template",)

    data_dpo_eval = data_dpo["test"].map(apply_chat_template,
                                    num_proc=cpu_count(),
                                    fn_kwargs={"tokenizer": tokenizer},
                                    remove_columns=['news'],
                                    desc="Applying chat template",)

    return data_dpo_train, data_dpo_eval


def estimate_token_num_dpo(tokenizer, data_dpo):

    prompt_length = int(max([len(tokenizer(x)["input_ids"]) for x in data_dpo["prompt"]]))
    max_seq_length_chosen = int(max([len(tokenizer(x["prompt"] + x["chosen"])["input_ids"]) for x in data_dpo]))
    max_seq_length_rejected = int(max([len(tokenizer(x["prompt"] + x["rejected"])["input_ids"]) for x in data_dpo]))
    max_seq_length = max(max_seq_length_chosen, max_seq_length_rejected)

    print(f"max prompt length: {prompt_length}")
    print(f"max prompt + chosen length: {max_seq_length}")  


def estimate_token_num_sft(tokenizer, data_path, task):

    dataset = load_sft_dataset(data_path, task)
    max_system_length = int(max([len(tokenizer(x["messages"][0]["content"])["input_ids"]) for x in dataset]))
    max_user_length = int(max([len(tokenizer(x["messages"][1]["content"])["input_ids"]) for x in dataset]))
    max_assistant_length = int(max([len(tokenizer(x["messages"][2]["content"])["input_ids"]) for x in dataset]))

    print(f"max system length: {max_system_length}")
    print(f"max user length: {max_user_length}")
    print(f"max assistant length: {max_assistant_length}")  

