from utils import *


def main():

    # os.environ["WANDB_DISABLED"] = "true"

    data_args, model_args, sftconfig_args, peft_args = read_args_sft()

    device_string = PartialState().process_index

    dtrain = load_sft_dataset(data_args.train_path, data_args.task)
    # deval = load_sft_dataset(data_args.eval_path, data_args.task)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_id)
    tokenizer.model_max_length = model_args.tokenizer_model_max_length
    tokenizer.padding_side = model_args.tokenizer_padding_side
    tokenizer.add_special_tokens({'pad_token': model_args.tokenizer_pad_token}) 

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_id,
        attn_implementation="flash_attention_2", # set this to True if your GPU supports it (Flash Attention drastically speeds up model computations)
        torch_dtype=torch.bfloat16,
        use_cache=False, # set to False as we're going to use gradient checkpointing
        device_map={'':device_string}, #"auto"
        quantization_config=quantization_config,
    )

    model, tokenizer = setup_chat_format(model, tokenizer)

    for index in random.sample(range(len(dtrain)), 3):
        print(f"Sample {index} of the processed training set:\n\n{dtrain[index]['messages'][0]['content']}\n\n{dtrain[index]['messages'][1]['content']}\n\n{dtrain[index]['messages'][2]['content']}\n\n")



    # import wandb
    # wandb.login(key="3815f35ed462426c8e328c8581af08e6b05dff5c")
    # run = wandb.init(project='mistral-rationale-stg1-sft-epoch12x', job_type="training", anonymous="allow", name="r256-alpha128")

    training_args = SFTConfig(

        max_seq_length=sftconfig_args.max_seq_length,
        packing=sftconfig_args.packing,
        max_grad_norm=sftconfig_args.max_grad_norm,
        bf16=sftconfig_args.bf16,
        tf32=sftconfig_args.tf32,
        # do_eval=sftconfig_args.do_eval,
        # eval_strategy=sftconfig_args.eval_strategy,
        gradient_accumulation_steps=sftconfig_args.gradient_accumulation_steps,
        gradient_checkpointing=sftconfig_args.gradient_checkpointing,
        gradient_checkpointing_kwargs=sftconfig_args.gradient_checkpointing_kwargs,
        learning_rate=sftconfig_args.learning_rate,
        log_level=sftconfig_args.log_level,
        logging_steps=sftconfig_args.logging_steps,
        logging_strategy=sftconfig_args.logging_strategy,
        lr_scheduler_type=sftconfig_args.lr_scheduler_type,
        max_steps=sftconfig_args.max_steps,
        num_train_epochs=sftconfig_args.num_train_epochs,
        output_dir=sftconfig_args.output_dir,
        overwrite_output_dir=sftconfig_args.overwrite_output_dir,
        # per_device_eval_batch_size=sftconfig_args.per_device_eval_batch_size,
        per_device_train_batch_size=sftconfig_args.per_device_train_batch_size,
        save_strategy=sftconfig_args.save_strategy,
        save_total_limit=sftconfig_args.save_total_limit,
        seed=sftconfig_args.seed,
        warmup_ratio=sftconfig_args.warmup_ratio,
        # load_best_model_at_end=sftconfig_args.load_best_model_at_end,
        dataset_kwargs=sftconfig_args.dataset_kwargs,
        report_to=sftconfig_args.report_to, 

    )


    peft_config = LoraConfig(

        r=peft_args.r,
        lora_alpha=peft_args.lora_alpha,
        lora_dropout=peft_args.lora_dropout,
        bias=peft_args.bias,
        task_type=peft_args.task_type,
        target_modules=peft_args.target_modules,

    )

    trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dtrain,
            # eval_dataset=deval,
            tokenizer=tokenizer,
            peft_config=peft_config,
        )

    trainer.train()

    # wandb.finish()


if __name__ == "__main__":
    main()


