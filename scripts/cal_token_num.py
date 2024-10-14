from utils import *

data_dpo = load_from_disk("results/models/numhg/data/data_dpo_ecnc192499.hf")
model_id = "results/models/numhg/mistral/stg1-sft-r64a32lr2e4ep3x-checkpoint-507-merged"
data_path = "/home/zhen/Documents/RMIT/NumEval/NumEval-Task3/vastai/1x6000ada/numeval_task3/data/dtrainECNC.json"
task = "rationale_hg_stg1_ecnc"

tokenizer = AutoTokenizer.from_pretrained(model_id)
data_dpo = apply_dpo_chat_template(data_dpo, tokenizer, task)

estimate_token_num_dpo(tokenizer, data_dpo)
# estimate_token_num_sft(tokenizer, data_path, task)