def get_wikipedia_pages(categories):
    """Retrieve Wikipedia pages from a list of categories and extract their content"""
    
    # Create a Wikipedia object
    wiki_wiki = wikipediaapi.Wikipedia('Gemma AI Assistant (gemma@example.com)', 'en')
    
    # Initialize lists to store explored categories and Wikipedia pages
    explored_categories = []
    wikipedia_pages = []

    # Iterate through each category
    print("- Processing Wikipedia categories:")
    for category_name in categories:
        print(f"\tExploring {category_name} on Wikipedia")
        
        # Get the Wikipedia page corresponding to the category
        category = wiki_wiki.page("Category:" + category_name)
        
        # Extract Wikipedia pages from the category and extend the list
        wikipedia_pages.extend(extract_wikipedia_pages(wiki_wiki, category_name))
        
        # Add the explored category to the list
        explored_categories.append(category_name)

    # Extract subcategories and remove duplicate categories
    categories_to_explore = [item.replace("Category:", "") for item in wikipedia_pages if "Category:" in item]
    wikipedia_pages = list(set([item for item in wikipedia_pages if "Category:" not in item]))
    
    # Explore subcategories recursively
    while categories_to_explore:
        category_name = categories_to_explore.pop()
        print(f"\tExploring {category_name} on Wikipedia")
        
        # Extract more references from the subcategory
        more_refs = extract_wikipedia_pages(wiki_wiki, category_name)

        # Iterate through the references
        for ref in more_refs:
            # Check if the reference is a category
            if "Category:" in ref:
                new_category = ref.replace("Category:", "")
                # Add the new category to the explored categories list
                if new_category not in explored_categories:
                    explored_categories.append(new_category)
            else:
                # Add the reference to the Wikipedia pages list
                if ref not in wikipedia_pages:
                    wikipedia_pages.append(ref)

    # Initialize a list to store extracted texts
    extracted_texts = []
    
    # Iterate through each Wikipedia page
    print("- Processing Wikipedia pages:")
    for page_title in tqdm(wikipedia_pages):
        try:
            # Make a request to the Wikipedia page
            page = wiki_wiki.page(page_title)

            # Check if the page summary does not contain certain keywords
            if "Biden" not in page.summary and "Trump" not in page.summary:
                # Append the page title and summary to the extracted texts list
                if len(page.summary) > len(page.title):
                    extracted_texts.append(page.title + " : " + clean_string(page.summary))

                # Iterate through the sections in the page
                for section in page.sections:
                    # Append the page title and section text to the extracted texts list
                    if len(section.text) > len(page.title):
                        extracted_texts.append(page.title + " : " + clean_string(section.text))
                        
        except Exception as e:
            print(f"Error processing page {page.title}: {e}")
                    
    # Return the extracted texts
    return extracted_texts