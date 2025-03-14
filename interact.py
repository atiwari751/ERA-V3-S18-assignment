import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_model_and_tokenizer():
    """Load the phi-2 model and tokenizer in 4-bit precision"""
    print("Loading phi-2 model and tokenizer in 4-bit precision...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/phi-2", 
        trust_remote_code=True
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
    
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters in 4-bit precision")
    return model, tokenizer

def interact_with_model(model, tokenizer):
    """Interactive mode to test the model in the terminal"""
    print("\n===== Model Interactive Mode =====")
    print("Type your prompts to test the model. Type 'exit' to quit.")
    
    model.eval()  # Set model to evaluation mode
    
    while True:
        # Get user input
        user_input = input("\nYour prompt: ")
        
        # Check if user wants to exit
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Exiting interactive mode.")
            break
        
        # Tokenize input
        inputs = tokenizer(user_input, return_tensors="pt").to(device)
        
        # Generate response
        with torch.no_grad():
            output = model.generate(
                inputs["input_ids"],
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode and print response
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        print("\nModel response:", response)

if __name__ == "__main__":
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Start interactive mode
    interact_with_model(model, tokenizer) 