from utils import *

data_path = "output/numhg/ecnc/mistral/rationales-stg1-dpo-r256a128lr2e6b08"
task = "rationale_hg_stg1_ecnc"
num_processes = 8
concat_output_files(data_path, task, num_processes)