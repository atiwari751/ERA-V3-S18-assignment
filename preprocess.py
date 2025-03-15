from datasets import Dataset
import pandas as pd

def format_instruction(row):
    """
    Format a conversation into the instruction format expected by Phi-2.
    
    Args:
        row: A row from the Open Assistant dataset
        
    Returns:
        Formatted text in instruction format
    """
    # For Open Assistant dataset, we need to extract the conversation
    # and format it properly for instruction fine-tuning
    
    # Create a simple instruction-response format
    if 'text' in row:
        return row['text']
    
    # For oasst1 dataset, create a simple format based on role
    if 'role' in row:
        if row['role'] == 'prompter':
            return f"Human: {row['text']}"
        elif row['role'] == 'assistant':
            return f"Assistant: {row['text']}"
    
    return row['text'] if 'text' in row else "No text available"

def preprocess_open_assistant_dataset(dataset):
    """
    Preprocess the Open Assistant dataset for fine-tuning.
    
    Args:
        dataset: The raw Open Assistant dataset
        
    Returns:
        Processed dataset ready for training
    """
    print("Preprocessing Open Assistant dataset...")
    
    # For simplicity, we'll just use the text field directly
    # This approach works with the basic SFTTrainer
    
    # If the dataset doesn't have a 'text' field, we'll create one
    if 'text' not in dataset.column_names:
        dataset = dataset.map(
            lambda x: {'text': format_instruction(x)},
            remove_columns=dataset.column_names
        )
    
    print(f"Processed dataset contains {len(dataset)} examples")
    print(f"Sample processed example: {dataset[0]}")
    
    return dataset 