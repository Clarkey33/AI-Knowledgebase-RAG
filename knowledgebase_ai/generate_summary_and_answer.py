#from knowledgebase_ai.add_indefinite_article import add_indefinite_article
#from knowledgebase_ai.get_embedding import get_embedding
#from knowledgebase_ai.clean_text import clean_text
import numpy as np
from nltk.tokenize import sent_tokenize
from typing import Tuple, List, Union
import re
from sentence_transformers import SentenceTransformer
from knowledgebase_ai.GemmaHF import GemmaHF
#from transformers import AutoTokenizer


def generate_summary_and_answer(query:str,
                                relevant_texts: List[str],
                                embedding_model: SentenceTransformer,
                                gemma_model = GemmaHF,
                                max_new_tokens: int=150,
                                temperature: float=0.3,
                                role: str ="expert") -> Tuple[str,List[str]]:
    
    """Generate an answer for a given question using context from a dataset
        tuple:(answer, cleaned_sources)
    """
    
    #Validate inputs
    if not relevant_texts or not isinstance(relevant_texts, list):
        return "I could not find relevant information to answer this question.", []
    
    #Debug logging
    print(f"Received {len(relevant_texts)} context chunks")

    #Handle duplicates and clean text
    seen = set()
    unique_texts=[]
    for text in relevant_texts:
        if not text or not isinstance(text,str):
            continue
        clean_text=text.strip()
        if clean_text not in seen:
            seen.add(clean_text)
            unique_texts.append(clean_text)
            if len(unique_texts) >= 3: #select top 3 only
                break
    
    if not unique_texts:
        return "No valid context found to answer this question.", []
    
    #Build prompt
    try:
        context = "\n\n".join(unique_texts[:3])

        prompt_template = """ [System]
        Role: {role}
        Instruction: Read the following context and provide a concise 1-2 sentences answer to the question.

        Context: {context}

        Question: {query}

        Answer:"""

        prompt = prompt_template.format(role=role,
                                        context = context,
                                        query=query)
    except Exception as e:
        print(f"Prompt generation failed: {e}")
        return "Error preparing the response.", []

    #generate model response
    try :
        response = gemma_model.generate_text(prompt=prompt,
                                             max_new_tokens=max_new_tokens,
                                             temperature=min(max(temperature,0.1),1.0),
                                             top_p=0.9,
                                             repetition_penalty=1.2)
        
        # Debug logging
        print(f"Raw response type: {type(response)}")
        print(f"Raw response: {response}")

        
       #process response
        answer = _extract_answer(response)
        answer = _postprocess_answer(answer) #Clean whitespace

        #Validate answer
        if not answer or len(answer) < 10:
            print(f"Invalide answer: {answer}")
            print(f"Full response:{response}")
            return "I couldn't generate a proper respnse.", []

        #clean sources
        clean_sources =[
            (text[:200] + "..." ) if len(text) > 200 else text
            for text in unique_texts]
            
        return answer, clean_sources      
    
    except Exception as e:
        print(f"Generation failed: {e}")
        return "I could not generate a proper response.", []
 

def _extract_answer(response: Union[str, List[str]]) -> str:
    '''extract answer from model response'''
    if not response:
        return ""
    
    #get text from response
    text = response[0] if isinstance(response, list) else str(response)

    #debug 
    print(f"text taken from response[0]: {text}")

    #manage whitespace
    text = ' '.join(text.split())

    #find answer marker
    if "Answer:" in text:
        answer_part = text.split("Answer:")[-1]

        #debug 
        print(f"answer_part: {answer_part}")

        #remove any remaining prompt parts
        for marker in ["[System]","Role:","Context:","Question:"]:
            answer_part=answer_part.split(marker)[0]
        return answer_part.strip()
    
    #debug statements
    print(f'answer_part after markers removed: {answer_part}')
    print(f"function returns: {text.strip()}")
    
    return text.strip()


        #return text.split("Answer:")[-1].split('\n')[0].strip()
    
    #Fallback
    #lines = [line.strip() for line in text.split("\n") if line.strip()]

    #take last non empty line thats not part of the prompt template
    #for line in reversed(line):
        #if line and not any(marker in line for marker in ["[System]","Role:","Context:","Question:"]):
            #return line
        
    #final fallback - first 2 sentences
    #sentences = sent_tokenize(text)
    #if len(sentences) >=2:
        #return " ".join(sentences[:2]).strip()
    #elif sentences:
        #return sentences[0].strip()
    
    #return text[:300].strip() #truncate if nothing else works

       
def _compress_context(query: str,
                      texts: List[str],
                      embedding_model: SentenceTransformer,
                      gemma_model: GemmaHF,
                      max_tokens: int=512,
                      top_n: int = 5)-> str:
    
    '''re rank chunks and selects most important parts'''

    #Re-rank by query similarity
    query_embed = embedding_model.encode(query)
    text_embeds = embedding_model.encode(texts)
    scores = np.dot(text_embeds, query_embed)

    ranked = [t for _, t in sorted(zip(scores, texts), reverse = True)]

    #Select top chunks that fit
    selected =[]
    current_length =0
    tokenizer = gemma_model.tokenizer

    for text in ranked[:top_n]:
        text_tokens = len(tokenizer.tokenize(text))
        if current_length + text_tokens > max_tokens:
            break
        selected.append(text)
        current_length += text_tokens

    #Fallback if no chunks selected

    if not selected and ranked:
        truncated = tokenizer.decode(tokenizer.encode(ranked[0],
                                                      max_length = max_tokens))
        selected.append(truncated)
    return "\n".join(selected)

def _postprocess_answer(answer:str) -> str:
    ''' Clean model output'''

    for marker in ["[System]","[Context]","[Question]","Answer:"]:
        answer = answer.split(marker)[0]

    #Normalize whitespace
    answer = re.sub(r"\s+", " ", answer).strip()

    #Handle ending
    if answer and answer[-1] not in {".","?","!"}:
        answer = answer.rstrip(".,!?") +  "."

    
    return answer








    