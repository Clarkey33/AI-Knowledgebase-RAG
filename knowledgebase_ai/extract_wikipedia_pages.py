import wikipediaapi
import pandas as pd
from tqdm import tqdm
import os


def get_wikipedia_pages(categories, max_pages = 100, existing_kb_path = None):
    """Retrieve Wikipedia pages from a list of categories and extract their content
    categories: list of Wikipedia topics i.e.: "Data Science", "Machine Learning"
    max_pages: maximum nuber of pages to pull
    existing_kb_path: path to existing knowledge base

    return dataframe of pulled content
    
    """
    
    # Initialize Api
    wiki = wikipediaapi.Wikipedia(user_agent='AI Assistant (aiassist@example.com)', 
                                  language='en')
    
    # get pages from categories in Wikipedia 
    pages = set()

    for category in categories:
        cat_page= wiki.page(f"Category:{category}")
        if cat_page.exists():
            pages.update(m.title for m in cat_page.categorymembers.values()
                         if not m.title.startswith("Catgory:"))
            
    #process pages
    content =[]
    for title in tqdm(list(pages)[:max_pages], desc="Processing pages"):
        page=wiki.page(title)
        if page.exists() and page.text:
            text = f"{title}: {page.summary}"
            content.append({"wikipedia_text": clean_text(text),
                            "source":"wikipedia"})
    print(f"processed {len(content) } pages")
            
    #create/update knowledge base
    if existing_kb_path and os.path.exists(existing_kb_path):
        existing = pd.read_csv(existing_kb_path)
        
        # Validate existing structure
        if 'wikipedia_text' not in existing.columns:
            raise ValueError("Existing CSV missing 'wikipedia_text' column")
        
        new_df=pd.concat([existing,pd.DataFrame(content)])
    else:
        new_df = pd.DataFrame(content)


    return new_df.drop_duplicates(subset=["wikipedia_text"])

def clean_text(text):
    import re
    text = re.sub(r'\s+', ' ',text).strip() #remove extra whitespace
    return text


    