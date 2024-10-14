from utils import *

def main ():
    
    args = read_args_inference()

    pipe = load_pipline(args.model_path, args.adapter_path) #adapter_path

    print(pipe.tokenizer.padding_side, pipe.tokenizer.truncation_side, pipe.tokenizer.pad_token)

    dataset_inference = prepare_dataset_inference(args.data_path, args.rationale_path, args.task, args.duplicates_multiple)
    test_inference(pipe, dataset_inference, args.max_new_tokens, args.sample_num, args.temperature, args.num_check)

    if Accelerator().num_processes == 1:
        while True:

            user_input = input("Do you want to continue? (yes/no): ").strip().lower()

            if user_input == "yes":
                print("Continuing the program...", "\n")
                new_tokens_list = bulk_inference(pipe, dataset_inference, args.max_new_tokens, args.temperature, args.batch_size, args.inference_num)
                save_inference(new_tokens_list, args.output_dir, args.task)
                break
            elif user_input == "no":
                break
            else:
                print("Invalid input. Please type 'yes' or 'no'.")
    else:
        new_tokens_list = bulk_inference(pipe, dataset_inference, args.max_new_tokens, args.temperature, args.batch_size, args.inference_num)
        save_inference(new_tokens_list, args.output_dir, args.task)
        
if __name__ == "__main__":
    main()
        



