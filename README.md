# Phi-2 Fine-tuned Assistant

## Overview

This project fine-tunes Microsoft's Phi-2 model (2.7B parameters) on the Open Assistant dataset to create a helpful, instruction-following assistant. The fine-tuned model maintains Phi-2's efficiency while improving its ability to follow instructions and engage in helpful dialogue.

## Features

* Efficient Model: Based on Microsoft's Phi-2 (2.7B parameters)
* Fine-tuned on Open Assistant: Trained on high-quality human feedback data
* Low Resource Requirements: Can run on consumer hardware with 4GB+ VRAM
* Interactive Chat Interface: Streamlit-based UI for easy interaction
* Token Streaming: Responses appear word-by-word for a natural experience

## Live Demo

Try the model on Hugging Face Spaces: Phi-2 Fine-tuned Assistant

## Technical Details

### Model Architecture

* Base model: Microsoft Phi-2 (2.7B parameters)
* Fine-tuning method: Low-Rank Adaptation (LoRA)
* LoRA parameters: 7,864,320 trainable parameters (0.28% of base model)
* LoRA configuration:
  *  Rank (r): 16
  * Alpha: 32
  * Target modules: Attention (q_proj, k_proj, v_proj, o_proj) and MLP (gate_proj, up_proj, down_proj)
* Quantization: 4-bit precision (NF4)
* Training dataset: Open Assistant (oasst1)

### Training Process

The model was fine-tuned using:
* LoRA rank: 16
* Learning rate: 2e-4
* Batch size: 4 (with gradient accumulation steps of 4)
* Training steps: 500
* Warmup ratio: 0.03

### Performance
The fine-tuned model shows improved performance in:

* Following complex instructions
* Maintaining helpful and safe responses
* Generating coherent and contextually appropriate text
