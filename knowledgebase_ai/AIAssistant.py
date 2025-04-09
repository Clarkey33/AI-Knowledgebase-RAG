import torch
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import faiss
import csv
from typing import Tuple, List
import time


import transformers
from transformers import (AutoModelForCausalLM, 
                          AutoTokenizer, 
                          BitsAndBytesConfig,
                         )
from sentence_transformers import SentenceTransformer
from knowledgebase_ai.generate_summary_and_answer import generate_summary_and_answer
from knowledgebase_ai.GemmaHF import GemmaHF

class AIAssistant():
    """An AI assistant that interacts with users by providing answers based on a provided knowledge base"""
    
    def __init__(self,gemma_model,embeddings_name="thenlper/gte-large", temperature=0.8, role="expert"):
       
        """Initialize the AI assistant with Gemma model and FAISS GPU support."""

        # Initialize attributes
        self.embeddings_name = embeddings_name
        self.knowledge_base = []
        self.temperature = temperature
        self.role = role
        self.max_retries = 3  
        self.threshold = 0.85  # Configurable threshold

        # Initialize Gemma model 
        self.gemma_model = gemma_model 
        

         # Initialize FAISS resources
        self.index = None
        self.res = None
        
        
        # Load the embedding model
        self.embedding_model = SentenceTransformer(self.embeddings_name) #.to(self.device)
        try:
            self.embedding_model = SentenceTransformer(embeddings_name)
            # Test embedding model works
            test_embed = self.embedding_model.encode("test")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize embedding model: {str(e)}")

   


    def store_knowledge_base(self, knowledge_base):
        """Store the knowledge base"""
        self.knowledge_base=knowledge_base
        
    def learn_knowledge_base(self, knowledge_base):
        """Store and index the knowledge based to be used by the assistant"""
        try:
            # Storing the knowledge base
            self.store_knowledge_base(knowledge_base)
            print(f'Knowledge base stored with {len(self.knowledge_base)} entries')

            #clear previous resources
            if hasattr(self,'index'):
                del self.index
            if hasattr(self, 'res'):
                del self.res
            torch.cuda.empty_cache()

            # Compute embeddings and move them to correct device
            print("Generating embeddings for the knowledge base...")
            self.embeddings = np.array([
                self.embedding_model.encode(text, convert_to_numpy=True)
                for text in tqdm(self.knowledge_base)
                ]).astype(np.float32)
        
            #verify embeddings
            if len(self.embeddings.shape) !=2:
                raise ValueError(f"Invalid embedding shape: {self.embeddings.shape}")
            print(f"Embeddings generated.\n Shape: {self.embeddings.shape}")

            #Normalize Embeddings for FAISS inner product
            print('Normalizing embeddings...')
            faiss.normalize_L2(self.embeddings)
            print("Embeddings normalized.")

            
            #Build FAISS index
            print("Building FAISS index...")
            dim = self.embeddings.shape[1] #get the dimensions of embeddings
            self.res = faiss.StandardGpuResources() #initialize GPU resources

            #configuration options
            config =faiss.GpuIndexFlatConfig()
            config.useFloat16 = False #more precise than float16
            config.device=0 #specify GPU device

            self.index = faiss.GpuIndexFlat(self.res,
                                            dim,
                                            faiss.METRIC_INNER_PRODUCT,
                                            config) #Create Faiss index on GPU (Inner Product Search)
        
            #add vectors in batches 
            batch_size = 10000
            for i in range(0, len(self.embeddings),batch_size):
                batch = self.embeddings[i:i+batch_size]
                self.index.add(batch)
            print(f'FAISS index built with {self.index.ntotal} vectors.')


            #save resources
            self._save_index_and_embeddings()
        except Exception as e:
            print(f'Failed to build knowledge base: {str(e)}')
            raise

    def _save_index_and_embeddings(self):
        """Save index and embeddings"""
        try:
            np.save("normalized_embeddings.npy",self.embeddings)

            #Convert Gpu index to CPU before saving
            cpu_index= faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index,"faiss_index.bin")
            print('Resources saved succesfully.')

        except Exception as e:
            print(f"Failed to save resources: {str(e)}")
            raise

      
    def load_knowledge_base(self, filename):
        """Load knowledge base from a CSV and return as a list of texts."""
        try:
            df = pd.read_csv(filename)
            if len(df.columns) == 0:
                raise ValueError("Empty CSV file")
            
            self.knowledge_base = df.iloc[:, 0].astype(str).tolist()
            print(f"Knowledge base loaded with {len(self.knowledge_base)} entries.")

            return self.knowledge_base 
        
        except Exception as e:
            print(f"Failed to load knowledge base: {str(e)}")
            raise

        

    
    def query(self, query: str, top_k: int=5, threshold: float=0.7, max_retries: int=3) -> Tuple[str,List[str]]:
        """Query the knowledge base and retrieve relevant answers."""
        

        #Validate inputs
        if not isinstance(query,str) or not query.strip():
            raise ValueError("Query must be a non-empty string")
        
        if not hasattr(self,'index') or self.index is None:
            raise RuntimeError("FAISS index not initialized")
        
        #Debug statments
        print(f"\n=== New Query ===")
        print(f"Query: {query}")
        print(f"Knowledge base size: {len(self.knowledge_base)}")
        
        #initialize variables
        relevant_texts= []
        answer=""
        last_error= None

        for attempt in range(max_retries):
            try:
                #generate query embedding
                query_embed = self.embedding_model.encode(query,
                                                          convert_to_numpy=True,
                                                          show_progress_bar=False
                                                          ).reshape(1,-1).astype(np.float32)
                
                #Verify embedding dimensions
                if query_embed.shape[1] != self.index.d:
                    raise ValueError(f"Embedding dim mismatch: Query {query_embed.shape[1]} vs Index {self.index.d}")
                
                #normalize query embeddings
                faiss.normalize_L2(query_embed)

                
                #FAISS search with GPU/CPU fallback
                try:
                    distances, indices = self.index.search(query_embed,top_k)

                except RuntimeError as e:

                    if "GPU" in str(e):
                        print("Falling back to CPU index")
                        cpu_index = faiss.index_gpu_to_cpu(self.index)
                        distances, indices = cpu_index.search(query_embed, top_k)

                    else:
                        raise

                print("\nTop matches before filtering")
                for i, (idx,score) in enumerate(zip(indices[0],distances[0])):
                    print(f"{i+1}. Index:{idx}, Score:{score:.3f}")

                valid_indices =[idx for i, idx in enumerate(indices[0])
                                if (0 <= idx < len(self.knowledge_base) and
                                    (distances[0][i] >= threshold))]
            
                print("\nValid indices after filtering:",valid_indices)

                relevant_texts =[self.knowledge_base[idx] for idx in valid_indices]

                print(f"\nRetrieved {len(relevant_texts)} context chunks")

                #Generate answer if context is available
                if relevant_texts:
                    answer = self._generate_answer(query,relevant_texts)
                return answer, relevant_texts
            
            except Exception as e:
                last_error = e
                print(f"Attempt {attempt +1} failed: {str(e)}")

                if attempt == max_retries -1:
                    raise RuntimeError(f'Query failed after{max_retries} attempts') from last_error
                time.sleep(1) #pause before retry


    def _generate_answer(self, query:str, contexts: List[str]) -> str:
        """generate answer with validation"""

        try:
            #Combine contexts with newline separation
            context_str= "\n---\n".join(contexts)

            #generate with gemma
            result, sources = generate_summary_and_answer(query=query,
                                                 relevant_texts=contexts,
                                                 embedding_model= self.embedding_model,
                                                 gemma_model = self.gemma_model,
                                                 temperature=self.temperature,
                                                 role = self.role)
                

            #debug output
            print(f"Generated answer:{result}")
            print(f"Generated answer type:{type(result)}")

            #Validate Output
            if not isinstance(result, str) or not result.strip():
                raise ValueError("Empty or invalid model output")
                
            return result

        except Exception as e:
            print(f'Answer generation failed: {str(e)}')
            return "I couldn't generate a proper response. Please try again."
        
                         
        
        
            
    def set_temperature(self, temperature):
        """Set the temperature (creativity) of the AI assistant."""
        self.temperature = temperature
        
    def set_role(self, role):
        """Define the answering style of the AI assistant."""
        self.role = role
        
    def load_embeddings(self, filename="embeddings.npy"):
        """Load the embeddings from disk and index them"""
        self.embeddings = np.load(filename).astype(np.float32)
        # Rebuild FAISS index
        #self.index_embeddings()

    def save_model(self, model_dir='saved_model_gemma2'):
        """Save the Gemma LLM model,tokenizer"""
        self.gemma_model.model.save_pretrained(model_dir)
        self.gemma_model.tokenizer.save_pretrained(model_dir)
        print(f'Model and dependencies saved to {model_dir}')
    
    def save_embedding_model(self,model_dir= 'saved_model_embedding'):
        # Save embedding model if supported
        if hasattr(self.embedding_model, 'save_pretrained'):
            self.embedding_model.save_pretrained(model_dir)
        print(f'Model and dependencies saved to {model_dir}')

    def save_faiss_index(self, index_file='faiss_index.bin'):
        """Save the FAISS index to disk."""
        # Transfer the index to CPU if it's GPU-based
        if isinstance(self.index, faiss.GpuIndexFlat):
            index_cpu = faiss.index_gpu_to_cpu(self.index)  # Move to CPU
        else:
            index_cpu = self.index  # Already CPU-based

        # Save the CPU-based index
        faiss.write_index(index_cpu, index_file)
        print(f"FAISS index saved to {index_file}")

    
    def load_model(self, model_dir='saved_model_gemma2'):
        """Load the Gemma model, tokenizer, and embedding model"""
        self.gemma_model= GemmaHF(model_name=model_dir) #GemmaHF wrapper
        #self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    def load_emb_model(self, model_dir ='saved_model_embedding'):
        # Load embedding model
        try:
            self.embedding_model = SentenceTransformer(model_dir)
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            self.embedding_model = None
    
        print(f'Model and dependencies loaded from: {model_dir}')

    def load_faiss_index(self, index_file='faiss_index.bin'):
        """Load the FAISS index from disk"""
        self.index = faiss.read_index(index_file)
        print(f"FAISS index loaded from {index_file}")