import torch
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import faiss
import wikipediaapi
from typing import Optional, List


import transformers
from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer, 
                          BitsAndBytesConfig,
                         )
from sentence_transformers import SentenceTransformer
from knowledgebase_ai.define_device import define_device

class GemmaHF():
    """Wrapper for the Transformers implementation of Gemma"""
    
    def __init__(self, model_name, max_seq_length=2048):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        
        # Initialize the model and tokenizer
        print("\nInitializing model:")
        self.device = define_device()
        self.model, self.tokenizer = self.initialize_model(self.model_name, self.device, self.max_seq_length)
        
        
    def initialize_model(self, model_name, device, max_seq_length):
        """Initialize a 4-bit quantized causal language model (LLM) and tokenizer with specified settings"""

        # Define the data type for computation
        compute_dtype = getattr(torch, "float16")

        # Define the configuration for quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )

        # Load the pre-trained model with quantization configuration
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            quantization_config=bnb_config,
        )

        # Load the tokenizer with specified device and max_seq_length
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            device_map=device,
            max_seq_length=max_seq_length
        )
        
        # Return the initialized model and tokenizer
        return model, tokenizer
    
    
    def generate_text(self,
                      prompt: str,
                      max_new_tokens: int,
                      temperature: float,
                      top_p: float=0.9,
                      repetition_penalty: float=1.2) -> List[str]:
        
        """Generate text using the instantiated tokenizer and model with specified settings"""

        print(f'max new tokens: {max_new_tokens}')
        print(f'temperature: {temperature}')

        # Total token limit
        total_token_limit = self.model.config.max_position_embeddings  
        print(f'total token limit: {total_token_limit}')

        #calculate max input length
        max_input_length = total_token_limit - (max_new_tokens if max_new_tokens is not None else 50)
        print(f'max input legnth: {max_input_length}')
    
        # Encode the prompt and convert to PyTorch tensor
        input_ids = self.tokenizer(prompt, return_tensors="pt", padding=False ,truncation= True, max_length =max_input_length).to(self.device)
        input_ids = input_ids.to(self.device)  # Move input to GPU
        
        
        #print(f'tokenized inputs: {input_ids}') #checking for correct generation
                
        # Calculate max tokens for generation
        input_length = input_ids['input_ids'].shape[1]
        print(f"input length:{input_length}")

        #calculate max tokens for generation
        max_tokens_for_generation = total_token_limit - input_length
        print(f'Max tokens for generation: {max_tokens_for_generation}')

        if max_tokens_for_generation <= 0:
            raise ValueError("Input sequence too long to allow for token generation!")

        # Generate text

        # Determine if sampling should be performed based on temperature
        do_sample = True if temperature > 0 else False

        # Check model's configuration
        print(f"Model max position embeddings: {self.model.config.max_position_embeddings}")
        print(f"Tokenizer max length: {self.tokenizer.model_max_length}")

        # Check input tensor dimensions
        print(f"Input IDs shape: {input_ids['input_ids'].shape}")
        print(f"Attention Mask shape: {input_ids['attention_mask'].shape}")



        # Generate text based on the input prompt
        try:
            outputs = self.model.generate(**input_ids, #unpack tokenized inputs
                                          max_new_tokens=max_tokens_for_generation,
                                          do_sample=do_sample,
                                          temperature=temperature,
                                          top_p= top_p,
                                          repetition_penalty=repetition_penalty
                                          )
        except Exception as e:
            print(f'Error during text generation: {e}')
            return []

        # Decode the generated output into text
        results = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        if not results or len(results) == 0:
            print("No results generated by the model.")
            
        # Return the list of generated text results
        return results