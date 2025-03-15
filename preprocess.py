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
    
    if "messages" in row:
        messages = row["messages"]
        conversation = ""
        
        for msg in messages:
            role = "Human: " if msg["role"] == "user" else "Assistant: "
            conversation += f"{role}{msg['content']}\n\n"
        
        return conversation.strip()
    
    # If the dataset structure is different, use a simpler approach
    if "text" in row:
        return row["text"]
    
    # Fallback for the basic structure in oasst1
    if "instruction" in row and "response" in row:
        return f"Human: {row['instruction']}\n\nAssistant: {row['response']}"
    
    return None

def preprocess_open_assistant_dataset(dataset):
    """
    Preprocess the Open Assistant dataset for fine-tuning.
    
    Args:
        dataset: The raw Open Assistant dataset
        
    Returns:
        Processed dataset ready for training
    """
    print("Preprocessing Open Assistant dataset...")
    
    # Convert to pandas for easier manipulation
    df = pd.DataFrame(dataset)
    
    # The oasst1 dataset has a tree structure, we need to extract conversations
    # For simplicity, we'll focus on the direct parent-child pairs
    
    # Extract text from the dataset
    if "text" not in df.columns:
        # Apply formatting function to create the text field
        formatted_texts = []
        for i, row in df.iterrows():
            formatted_text = format_instruction(row)
            if formatted_text:
                formatted_texts.append({"text": formatted_text})
        
        # Create a new dataset with the formatted text
        processed_dataset = Dataset.from_pandas(pd.DataFrame(formatted_texts))
    else:
        # If text field already exists, use it directly
        processed_dataset = dataset
    
    print(f"Processed dataset contains {len(processed_dataset)} examples")
    print(f"Sample processed example: {processed_dataset[0]}")
    
    return processed_dataset 