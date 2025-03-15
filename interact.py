import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from peft import PeftModel
import threading
import sys
import time

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_finetuned_model_and_tokenizer(model_path="./final_model"):
    """Load the fine-tuned phi-2 model and tokenizer in 4-bit precision"""
    print(f"Loading fine-tuned model from {model_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/phi-2", 
        trust_remote_code=True
    )
    # Set pad_token to a different value than eos_token to fix attention mask issue
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.unk_token
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # Load base model with 4-bit quantization
    base_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto"  # Automatically distribute across available GPUs
    )
    
    # Load the fine-tuned LoRA adapter
    model = PeftModel.from_pretrained(
        base_model,
        model_path,
        device_map="auto"
    )
    
    print(f"Fine-tuned model loaded successfully!")
    return model, tokenizer

def interact_with_model(model, tokenizer):
    """Interactive mode to test the model in the terminal with streaming output"""
    print("\n===== Fine-tuned Model Interactive Mode =====")
    print("Type your prompts to test the model. Type 'exit' to quit.")
    
    model.eval()  # Set model to evaluation mode
    conversation_history = ""
    
    while True:
        # Get user input
        user_input = input("\nYour prompt: ")
        
        # Check if user wants to exit
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Exiting interactive mode.")
            break
        
        # Format input with proper instruction format
        formatted_input = f"Human: {user_input}\n\nAssistant:"
        
        # Add to conversation history if needed
        if conversation_history:
            full_prompt = f"{conversation_history}\n\n{formatted_input}"
        else:
            full_prompt = formatted_input
        
        # Tokenize input with explicit attention mask
        inputs = tokenizer(full_prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Create a streamer for token-by-token generation
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Set generation parameters
        generation_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "max_new_tokens": 500,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "streamer": streamer,
        }
        
        # Start generation in a separate thread
        print("\nAssistant: ", end="", flush=True)
        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Print tokens as they're generated
        generated_text = ""
        for new_text in streamer:
            # Check if the model is trying to start a new turn
            if "Human:" in new_text or "\nHuman:" in generated_text + new_text:
                # Stop generation if model tries to create a new human turn
                break
            if "Assistant:" in new_text and generated_text:
                # Stop if model tries to create a new assistant turn
                break
            
            print(new_text, end="", flush=True)
            generated_text += new_text
            # Add a small delay to make the streaming more visible
            time.sleep(0.01)
        
        # Update conversation history for context
        conversation_history = f"{full_prompt} {generated_text.strip()}"

if __name__ == "__main__":
    # Load fine-tuned model and tokenizer
    model, tokenizer = load_finetuned_model_and_tokenizer()
    
    # Start interactive mode
    interact_with_model(model, tokenizer) 