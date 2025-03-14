import torch
from peft import LoraConfig, get_peft_model

def find_target_modules(model):
    """Find all linear modules in the model to identify potential LoRA targets"""
    target_modules = []
    
    # Iterate through named modules
    for name, module in model.named_modules():
        # Check if it's a Linear module
        if isinstance(module, torch.nn.Linear):
            target_modules.append(name)
    
    print("Potential target modules for LoRA:")
    for name in target_modules:
        print(f"  - {name}")
    
    return target_modules

def setup_lora_for_model(model):
    """Configure and apply LoRA adapters to the model"""
    print("Setting up LoRA for fine-tuning...")
    
    # Optionally: Uncomment to find all potential target modules
    # find_target_modules(model)
    
    # For phi-2, these are the correct target modules
    # If using a different model, you may need to adjust these
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention modules
        "gate_proj", "up_proj", "down_proj"      # MLP modules
    ]
    
    # Define LoRA configuration
    lora_config = LoraConfig(
        r=16,                     # Rank dimension
        lora_alpha=32,            # Alpha parameter for LoRA scaling
        target_modules=target_modules,
        lora_dropout=0.05,        # Dropout probability for LoRA layers
        bias="none",              # Don't train bias parameters
        task_type="CAUSAL_LM"     # Task type for causal language modeling
    )
    
    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters info
    model.print_trainable_parameters()
    
    return model 