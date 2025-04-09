from PyPDF2 import PdfReader
import pandas as pd
import os

def pdf_to_knowledge_base(file_path):
    """Extracts text from a PDF file and returns it as a list of text chunks """
    print(f"Received file path type: {type(file_path)}, value: {file_path}")

    try:
        if not isinstance(file_path,str): #check that file_path is a string
            raise ValueError("file path must be a string")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        reader = PdfReader(file_path)
        pages = []
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                if text and text.strip():
                    pages.append(text.strip())
                    #print(f"Processed page{i+1}")
                    
            except Exception as e:
                print(f"Warning: Could not extract page{i+1}: {str(e)}")
                continue
        
        print(f"Succesfully extracted {len(pages)} pages")
        return pages
        
    except Exception as e:
            print(f"Error processing PDF:{str(e)}")
            return []
            

def txt_to_knowledge_base(file_path):
    """Extracts text from a text file and returns it as a list of chunks."""
    print(f"Received file path type: {type(file_path)}, value: {file_path}")

    try:
        if not isinstance(file_path, str):
            raise ValueError("File path must be a string")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as file:
            text=file.read()
            return [page.strip() for page in text.split('\n\n') if page.strip()]
        
    except Exception as e:
        print(f"Error processing text file: {str(e)}")
        return []


def update_knowledge_base_with_data(file_path,kb_file_path="./wikipedia_data_science_kb"):
    """Updates the knowledge base with text from PDFs or text files or Wikipedia API."""
    print(f"Processing {file_path}...")
    try:
        
        #get new knowledge
        new_knowledge=[] #initialize empty list to store new knowledge

        if file_path.lower().endswith(".pdf"): #check if file pdf
            new_knowledge= pdf_to_knowledge_base(file_path)
            print(f"New knowledge updated with {len(new_knowledge)} pages")

        elif file_path.lower().endswith(".txt"):
            new_knowledge = txt_to_knowledge_base(file_path)   
        else:
            print('Error: Only pdf or text files supported')
            return []

        if not new_knowledge:
            print("No new knowledge extracted")
            return [] #return empty list


        #load existing knowledge base
        if os.path.exists(kb_file_path):
            existing_kb = pd.read_csv(kb_file_path)
            old_knowledge = set(existing_kb['wikipedia_text'].astype(str))
        else:
            old_knowledge=set()
            existing_kb = pd.DataFrame(columns=['wikipedia_text'])

        #list of new knowledge; no duplicates
        updated_knowledge = [text for text in new_knowledge 
                             if text not in old_knowledge]

        if not updated_knowledge:
            print("No new knowledge to add")
            return []
        
        #save updated kb
        updated_kb_df=pd.concat([existing_kb,
                                 pd.DataFrame({'wikipedia_text':updated_knowledge})],
                                 ignore_index=True)

        updated_kb_df.to_csv('updated_kb.csv',index=False)
        print(f"Succesfully updated knowledge base with {len(updated_knowledge)} new entries.")

        return updated_kb_df

    except Exception as e:
        print(f"Error updating knowledge base: {str(e)}")
        return []


#if __name__ == "__main__":
    #result = update_knowledge_base_with_data("hr_manual.pdf", "./AML3406_Capstone/wikipedia_data_science_kb.csv")
    #if result:
        #print("Update successful")
    #else:
        #print("Update failed")

    


            

