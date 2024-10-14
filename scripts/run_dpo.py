from utils import *

def main():

    dpo_args = read_args_dpo()

    data_dpo = load_from_disk(dpo_args.data_path)
    data_dpo = data_dpo.shuffle(seed=42)
    data_dpo = data_dpo.train_test_split(test_size=0.02, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(dpo_args.model_id)
    tokenizer.model_max_length = 2048
    tokenizer.padding_side = 'left' # to prevent errors with FA
    tokenizer.truncation_side = 'left' # to prevent cutting off last generation
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})


    data_dpo_train, data_dpo_eval = apply_dpo_chat_template(data_dpo, tokenizer, dpo_args.task)

    for index in random.sample(range(len(data_dpo_train)), 5):
        print(f"Sample {index} of the processed training set:\n\n{data_dpo_train[index]['prompt']}{data_dpo_train[index]['chosen']}{data_dpo_train[index]['rejected']}")
        

    #### option 2: merge the adapter from part 1 to based model first. No need to initialize a ref_model in this case.
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.bfloat16,
    )

    peft_config = LoraConfig(
        r=dpo_args.r,
        lora_alpha=dpo_args.lora_alpha,
        lora_dropout=dpo_args.lora_dropout,
        bias=dpo_args.bias,
        task_type=dpo_args.task_type,
        target_modules=dpo_args.target_modules,
    )

    device_string = PartialState().process_index

    model = AutoModelForCausalLM.from_pretrained(
        dpo_args.model_id, 
        torch_dtype=torch.bfloat16, 
        quantization_config=quantization_config, 
        attn_implementation="flash_attention_2", 
        device_map={'':device_string}, 
        use_cache=False
        )

    # wandb.login(key="3815f35ed462426c8e328c8581af08e6b05dff5c")
    # run = wandb.init(project='mistral_rationale_hg_stg1_dpo', job_type="training", anonymous="allow", name="r512-alpha256-b03-dev12301")

    training_args = DPOConfig(

        max_prompt_length=dpo_args.max_prompt_length,
        max_length=dpo_args.max_length,    
        beta=dpo_args.beta,
        remove_unused_columns=dpo_args.remove_unused_columns,
        do_eval=dpo_args.do_eval,
        eval_strategy=dpo_args.eval_strategy,
        eval_steps=dpo_args.eval_steps,
        per_device_eval_batch_size=dpo_args.per_device_eval_batch_size,
        per_device_train_batch_size=dpo_args.per_device_train_batch_size,
        gradient_accumulation_steps=dpo_args.gradient_accumulation_steps,
        gradient_checkpointing=dpo_args.gradient_checkpointing,
        gradient_checkpointing_kwargs=dpo_args.gradient_checkpointing_kwargs,
        learning_rate=dpo_args.learning_rate,
        max_grad_norm=dpo_args.max_grad_norm,
        lr_scheduler_type=dpo_args.lr_scheduler_type,
        max_steps=dpo_args.max_steps,
        num_train_epochs=dpo_args.num_train_epochs,
        save_steps=dpo_args.save_steps,
        save_strategy=dpo_args.save_strategy,
        save_total_limit=dpo_args.save_total_limit,
        log_level=dpo_args.log_level,
        logging_steps=dpo_args.logging_steps,
        output_dir=dpo_args.output_dir,
        optim=dpo_args.optim,
        warmup_ratio=dpo_args.warmup_ratio,
        bf16=dpo_args.bf16,
        seed=dpo_args.seed,
        load_best_model_at_end=dpo_args.load_best_model_at_end,
        report_to=dpo_args.report_to,

    )

    # Create DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        args=training_args,
        train_dataset=data_dpo_train,
        eval_dataset=data_dpo_eval,
        tokenizer=tokenizer,
        peft_config=peft_config,
        loss_type="sigmoid",

    )


    dpo_trainer.train()



if __name__ == "__main__":
    main()

