import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer
from peft import PeftModel
import threading

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_tokenizer(model_path="./final_model"):
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
    
    model.eval()  # Set model to evaluation mode
    print(f"Fine-tuned model loaded successfully!")
    return model, tokenizer

def format_chat_history(messages):
    """Format the chat history into a prompt for the model"""
    formatted_prompt = ""
    
    for message in messages:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            formatted_prompt += f"Human: {content}\n\n"
        elif role == "assistant":
            formatted_prompt += f"Assistant: {content}\n\n"
    
    # Add the final assistant prompt
    formatted_prompt += "Assistant:"
    
    return formatted_prompt

def generate_response(model, tokenizer, messages):
    """Generate a streaming response from the model based on chat history"""
    # Format the conversation history
    prompt = format_chat_history(messages)
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
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
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Stream tokens as they're generated
    generated_text = ""
    for new_text in streamer:
        # Check if the model is trying to start a new turn
        if "Human:" in new_text or "\nHuman:" in generated_text + new_text:
            # Stop generation if model tries to create a new human turn
            break
        if "Assistant:" in new_text and generated_text:
            # Stop if model tries to create a new assistant turn
            break
        
        yield new_text
        generated_text += new_text 