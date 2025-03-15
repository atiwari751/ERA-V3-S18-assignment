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
    Preprocess the Open Assistant dataset for fine-tuning with TRL 0.15.2.
    
    Args:
        dataset: The raw Open Assistant dataset
        
    Returns:
        Processed dataset ready for training
    """
    print("Preprocessing Open Assistant dataset...")
    
    # Extract only the necessary fields for training
    cleaned_data = []
    
    for example in dataset:
        # Extract only the text field and create a clean example
        if 'text' in example and example['text'] and isinstance(example['text'], str):
            # Format based on role if available
            if 'role' in example:
                if example['role'] == 'prompter':
                    text = f"Human: {example['text']}"
                elif example['role'] == 'assistant':
                    text = f"Assistant: {example['text']}"
                else:
                    text = example['text']
            else:
                text = example['text']
                
            # Add to cleaned data
            cleaned_data.append({"text": text})
    
    # Create a new dataset with only the text field
    processed_dataset = Dataset.from_list(cleaned_data)
    
    print(f"Processed dataset contains {len(processed_dataset)} examples")
    if len(processed_dataset) > 0:
        print(f"Sample processed example: {processed_dataset[0]}")
    
    return processed_dataset 