#streamlit test
import streamlit as st
import requests
import time
from tempfile import NamedTemporaryFile

#page selection in sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Chat","Evaluation"])

if page == "Chat":
    
    # Configuration
    BACKEND_URL = "http://localhost:8000"

    #set title for GUI
    st.title('HR Assistant: Knowledge Base AI')

    #initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages=[]

    #Display chat message from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Sources"):
                    for src in message["sources"]:
                        st.caption(src) #display sources
        
    #Accept User input
    if prompt:= st.chat_input("Ask about HR policies"):
        #add user message to chat history
        st.session_state.messages.append({"role": "user",
                                          "content":prompt}
                                          )
        
        #display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        #get response from backend endpoint
        with st.chat_message("assistant"):
            try:
                #call FastAPI endpoint
                response = requests.post(
                    f"{BACKEND_URL}/query",
                    json={"query": prompt},
                    timeout=10)
                
                #check for HTTP errors
                response.raise_for_status()
                data = response.json()

                #check if response exists
                if "response" not in data:
                    raise ValueError("API response missing 'response' field")
                
                #stream the response
                message_placeholder=st.empty()
                full_response = ""

                #simulate streaming by words
                response_text = data.get("response", "")
                for word in response_text.split():
                    full_response += word + " "
                    message_placeholder.markdown(full_response + "▌") #block is for cursor effect
                    time.sleep(0.05) #small delay for streaming effect

                #Final message without cursor
                message_placeholder.markdown(full_response)

                #display response
                #st.markdown(data["response"])

                #show sources if available
                if data.get("sources"):
                    with st.expander("Sources"): #drop down menu with sources
                        for src in data["sources"]:
                            st.caption(src)

                #add to history with sources
                st.session_state.messages.append({
                    "role":"assistant",
                    "content": data["response"],
                    "sources": data.get("sources",[])
                })

            except requests.exceptions.RequestException as e:
                st.error(f"API Error: {str(e)}")
            except KeyError:
                st.error("Unexpected response format from API")
            except Exception as e:
                st.error(f"Error: {str(e)}")  

    #create side panel wid admin options to update the knowledge base
    management_expander = st.sidebar.expander("⚙️ Admin Panel", expanded=False)

    with management_expander:
        st.write("Knowledge Base Management")
    
        tab_1, tab_2 = st.tabs(['update knowledge base', 'get wikipedia pages'])  

        with tab_1:
            st.header("Upload Documnet")
            upload_file = st.file_uploader("Upload Document",
                                           type=["pdf","txt"])
    
            if upload_file is not None: #if file exists show metrics below
                col_1, col_2 = st.columns(2)
                col_1.metric("filename:", upload_file.name) #file name
                col_2.metric("file size", f"{round(len(upload_file.getvalue())/1024**2,1)}MB ") #file size

                if st.button("Process and Update"):
                    with st.spinner("Updating Knowledge base..."):
                        try:
                            file_particulars = {"file": (upload_file.name,upload_file.getvalue())}

                            response = requests.post(
                                f"{BACKEND_URL}/update_knowledge_base",
                                files=file_particulars,
                                timeout=60
                                )
                            response.raise_for_status()

                            result = response.json()

                            st.success(f"Succesfully updated. Added {result.get('new_entries',0)} new entries")

                            st.json(result)
                        except requests.exceptions.RequestException as e:
                            st.error(f" Update failed: {str(e)}")
                        except Exception as e:
                            st.error(f"Unexpected error: {str(e)}")

            else:
                st.warning("Please select a file first")

        with tab_2:
            st.header("Update Sources")

            with st.form('wiki_update_form'):
                #input fields
                categories = st.text_input(
                    "Enter Wikiepedia categories (comma separated)",
                    " "
                    )
            
                max_pages = st.slider(
                    "Maximum pages to pull",
                    min_value = 1,
                    max_value=100,
                    value=10,
                    help=" Limits number of pages retrieved"
                    )

                security_token = st.text_input(
                    "Security Token",
                    type= "password",
                    help = "Required for knowledge base updates"
                    )

                #form submission
                if st.form_submit_button("Update from Wikipedia"):
                    with st.spinner(f"Fetching {max_pages} Wikipedia pages..."):
                        try:

                            #prepare request
                            payload={"token": security_token,
                                     "max_pages": max_pages,
                                    "categories": [cat.strip() for cat in categories.split(",")
                                                   if cat.strip()]
                                                   }
                        
                            #connect to backend
                            response = requests.post(f"{BACKEND_URL}/admin/update_from_wikipedia",
                                                     json=payload,
                                                     timeout= 3600 #60 minute timout for fetching files
                            )

                            response.raise_for_status()

                            #Display results
                            result = response.json()
                            st.success(f" Successfully updated. Added{result['new_entries']} new entries")

                        #show fetched pages
                        #with st.expander("View Updated pages"):
                            #st.json(result)

                        except requests.exceptions.RequestException as e:
                            st.error(f"Connection failed:{str(e)}")
                        except Exception as e:
                            st.error(f"Unexpected error: {str(e)}")
                        
else:
    #import and run evaluation page
    from model_evaluator import show
    show()


        
















                



