from transformers import TrainingArguments

def get_training_arguments(output_dir="./results"):
    """
    Configure and return the training arguments for fine-tuning.
    
    Args:
        output_dir: Directory where model checkpoints and logs will be saved
        
    Returns:
        TrainingArguments object with all hyperparameters set
    """
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_steps=10,
        logging_steps=1,
        learning_rate=2e-4,
        fp16=True,
        max_grad_norm=0.3,
        max_steps=500,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        gradient_checkpointing=True,
    )

def get_sft_config():
    """
    Return configuration parameters for the Supervised Fine-Tuning process.
    
    Returns:
        Dictionary with SFT-specific parameters
    """
    return {
        "max_seq_length": 512,
        "dataset_text_field": "text",  # Will be updated by preprocessing
    } 