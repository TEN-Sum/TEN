from utils import merge_adapter

adapter_path = "results/models/xsum/ecnc/llama3/stg1-sft-r128a64lr8e4ep3x/checkpoint-291"
output_path = "results/models/xsum/ecnc/llama3/stg1-sft-r128a64lr8e4ep3x-checkpoint-291-merged"
merge_adapter(adapter_path, output_path)