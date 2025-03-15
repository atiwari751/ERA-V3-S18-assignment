import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import os
from peft import prepare_model_for_kbit_training
from lora_config import setup_lora_for_model
from training_config import get_training_arguments, get_sft_config
from preprocess import preprocess_open_assistant_dataset
from trl import SFTTrainer

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_model_and_tokenizer():
    """Load the phi-2 model and tokenizer in 4-bit precision"""
    print("Loading phi-2 model and tokenizer in 4-bit precision...")
    
    # Get SFT configuration to access max_seq_length
    sft_config = get_sft_config()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/phi-2", 
        trust_remote_code=True,
        model_max_length=sft_config["max_seq_length"]  # Set max length here
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # Load model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto"  # Automatically distribute across available GPUs
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters in 4-bit precision")
    return model, tokenizer

def load_open_assistant_dataset():
    """Load the Open Assistant dataset"""
    print("Loading Open Assistant dataset...")
    
    # Load a subset of the Open Assistant dataset
    dataset = load_dataset("OpenAssistant/oasst1", split="train")
    
    # Print dataset info
    print(f"Dataset loaded with {len(dataset)} examples")
    #print(f"Sample example: {dataset[0]}")
    
    return dataset

def train_model(model, tokenizer, dataset):
    """Set up and run the training loop"""
    print("Setting up training...")
    
    # Get training arguments
    training_args = get_training_arguments()
    
    # Create SFT Trainer with minimal parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        tokenizer=tokenizer,  # This will show a deprecation warning but still works
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    print("Training complete! Saving model...")
    trainer.save_model("./final_model")
    
    return trainer

if __name__ == "__main__":
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Apply LoRA to the model
    model = setup_lora_for_model(model)
    
    # Load dataset
    raw_dataset = load_open_assistant_dataset()
    
    # Preprocess the dataset
    processed_dataset = preprocess_open_assistant_dataset(raw_dataset)
    
    # Train the model
    trainer = train_model(model, tokenizer, processed_dataset)
    
    print("Fine-tuning process completed successfully!")
