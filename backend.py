
""" 
Process flow

1. load libraries

2. load saved components into backend script; verify correct resources loaded

3. create endpoint in fastapi() app (backend) to enable queries

"""

# load libraries
import numpy as np
import pandas as pd
import time
from typing import Optional, List, ClassVar
import os
import sys
import traceback
import faiss
from knowledgebase_ai.AIAssistant import AIAssistant
from knowledgebase_ai.GemmaHF import GemmaHF
from knowledgebase_ai.update_knowledge_base import update_knowledge_base_with_data
from knowledgebase_ai.extract_wikipedia_pages import get_wikipedia_pages

from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

from typing import Union
from fastapi import FastAPI, HTTPException, Depends
from fastapi import UploadFile, File
from pydantic import BaseModel, Field, field_validator
from contextlib import asynccontextmanager
import logging
from logging.config import dictConfig
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import tempfile
from werkzeug.utils import secure_filename
import shutil


# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

#Configuration 
#initialize paths to saved resources
model_dir= "saved_model_gemma2"
model_dir_emb = "saved_model_embedding"
model_id="google/gemma-2b-it"
emb_id = "thenlper/gte-large"
embeddings_file = "normalized_embeddings.npy"
faiss_index_file = "faiss_index.bin"
knowledge_base_file = "wikipedia_data_science_kb.csv"
max_retries = 3
timeout = 30 #seconds
assistant = None




@app.on_event("startup")
def startup_event():
    global assistant
    try:
        logger.info("Initializing AI assistant...")

        #check for resources
        use_existing = all([os.path.exists(model_dir),
                            os.path.exists(embeddings_file),
                            os.path.exists(faiss_index_file)])
    
        #initialize models with retries
        for attempt in range(max_retries):
            try:
                if use_existing:
                    logging.info("Loading existing models...")
                    assistant =AIAssistant(gemma_model=None,
                                           embeddings_name=emb_id)
                    assistant.load_model(model_dir=model_dir)
                    assistant.load_emb_model(model_dir=model_dir_emb)
                    assistant.load_embeddings(embeddings_file)
                    assistant.load_faiss_index(faiss_index_file)

                else:
                    logging.info("Initializing new models...")
                    gemma_model = GemmaHF(model_name=model_id)
                    assistant = AIAssistant(gemma_model=gemma_model,
                                            embeddings_name= emb_id
                                            )
        
                    #save new models
                    assistant.save_model(model_dir)
                    assistant.save_embedding_model()
                    

                #load knowledge base
                logging.info("loading knowledge base...")
                knowledge_base = assistant.load_knowledge_base(knowledge_base_file)

                if not use_existing:
                    logging.info("Building new FAISS index...")
                    assistant.learn_knowledge_base(knowledge_base)
                    assistant._save_index_and_embeddings()

                    logger.info("Initialization complete.")
                    break #Success

            except Exception as e:
                if attempt == max_retries -1:
                    raise
                logger.warning(f"Attempts {attempt + 1} failed: {str(e)}")
                time.sleep(2)

    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Service initialization failed")



@app.get("/")
def root():
    return {"status": "ready",
            "model": model_id,
            "embedding_model": emb_id
            }

class QueryRequest(BaseModel):
    query: str 
    #threshold:float= Field(0.7, ge=0.1, le=1.0)

#intialize rouge score
rouge_metric = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer= True)

@app.post("/query")
async def query_ai(request: QueryRequest):
    try:
        response, sources = assistant.query(request.query,
                                            threshold=0.7)
        
        # Convert list of sources to a single string
        reference_text = " ".join(sources[:3]) 

        #tokenize for Bleu calculation
        #hypothesis_tokens = nltk.word_tokenize(response.lower())
        #reference_tokens = [nltk.word_tokenize(ref.lower()) for ref in sources[:3]]

        #calculaye scores
        #rouge_scores = rouge_metric.score(reference_text,response)
        #print(f"Rouge Scores are: {rouge_scores}")

        #bleu_score = sentence_bleu(reference_tokens,
                                   #hypothesis_tokens,
                                   #smoothing_function = SmoothingFunction().method1
                                   #)
        
        #print(f"Bleu Score is: {round(bleu_score,4)}")
                                   

        return {"response": response, 
                "sources":sources[:3],
                }
    except ValueError as e:
        raise HTTPException(status_code =400,
                            detail = f"Invalid query parameters:{str(e)}")
    except Exception as e:
        logger.error(f"Query faile: {traceback.format_exc()}")
        raise HTTPException(
            status_code =500,
            detail ="Internal server error"
        )
    

class UpdateKBRequest(BaseModel):
    description: Optional[str] = "Knowledge base update"

#add endpoint
@app.post("/update_knowledge_base")
async def update_knowledge_base(file: UploadFile= File(...)):
    '''Updates the knowledge base with new content from uploaded files
        Accepts: PDF or TXT files
    '''
    #temp_path= None
    temp_dir = tempfile.mkdtemp()
    try:
        #validate and secure file name
        filename = secure_filename(file.filename)
        temp_path = os.path.join(temp_dir, filename)
        
        
        #confirm file type 
        #if not file.filename.lower().endswith(('.pdf','.txt')):
            #raise HTTPException(status_code=400, detail="Only PDF/TXT files allowed")
    
        #save file temporarily
        #temp_path = f"temp_{int(time.time())}_{file.filename}"
        with open(temp_path, "wb") as f:
            while content  := await file.read(1024*1024):  #reads data in 1MB chunks
                f.write(content)

        #process file
        result = update_knowledge_base_with_data(file_path=temp_path,
                                                 kb_file_path=knowledge_base_file)
        
        #rebuild faiss index if update successful
        if isinstance(result,pd.DataFrame) and not result.empty:
            logger.info("Rebuilding FAISS index with updated knowledge base.")
            result.to_csv(knowledge_base_file, index=False)
            assistant.load_knowledge_base(knowledge_base_file)
            assistant.learn_knowledge_base(assistant.knowledge_base)
            assistant._save_index_and_embeddings()
        return {"status": "success", "new_entries": len(result)}
    
    except Exception as e:
        logger.error(f"Knowledeg base update failed: {str(e)}")
        raise HTTPException(status_code =400, detail=str(e))
    
    finally:
        #clean temp file
        shutil.rmtree(temp_dir, ignore_errors=True)
        #if temp_path in locals() and os.path.exists(temp_path):
            #os.remove(temp_path)


class WikipeidaUpdateRequest(BaseModel):
    '''Manual update endpoint for authorized users'''

    token: str #security token
    max_pages: int = 100
    categories: list[str]

    #class variable(not a field)
    CONFIG_TOKEN: ClassVar[str]  = "1234"

    @field_validator('token')
    def validate_token(cls,v:str)-> str:
        """Verify provided token matches config"""
        if v != cls.CONFIG_TOKEN:
            raise ValueError("Invalid authorization token")
        return v

    @field_validator('categories')
    def validate_categories(cls, v: list[str]) -> list[str]:
        """Ensure at least one category is provided"""
        if not v:
            raise ValueError(" Must provide at least one category")
        return v
    
@app.post("/admin/update_from_wikipedia")
async def update_from_wikipedia(request: WikipeidaUpdateRequest):
    #if request.token != WikipeidaUpdateRequest.CONFIG_TOKEN:
        #raise HTTPException(status_coe =403, detail = 'Invalide update token')
    
    
    try:
        kb_df= get_wikipedia_pages(categories=request.categories,
                                   max_pages=request.max_pages,
                                   existing_kb_path= knowledge_base_file)
        
        kb_df.to_csv(knowledge_base_file, index=False)

        #rebuild FAISS index
        assistant.load_knowledge_base(knowledge_base_file)
        assistant.learn_knowledge_base(assistant.knowledge_base)

        return{"status":"success", "new_entries": len(kb_df)}
    except Exception as e:
        raise HTTPException(status_code =400, detail=str(e))


#Utility FUnctions

async def run_with_timeout(func, *args, timeout:int):
    #Run function with a timeout protection
    import asyncio
    try:
        return await asyncio.wait_for(asyncio.to_thread(func,*args),
                                      timeout=timeout
                                      )
    except asyncio.TimeoutError:
        raise TimeoutError(f"Operation exceeded {timeout} seconds")
    
if __name__ == "__main__":
    import uvicorn
    


        




