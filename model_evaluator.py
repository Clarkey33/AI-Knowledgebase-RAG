import streamlit as st
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
import numpy as np
import requests

def evaluate_output(predicted, reference):
    """calculate evaluation metrics"""
    
    #intialize rouge
    rouge_scores = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'],
                                            use_stemmer=True)
    smoothing =SmoothingFunction().method1

    #Tokenize for BLEU Score
    pred_tokens = nltk.word_tokenize(predicted.lower()) #generated answer tokens
    ref_tokens = [nltk.word_tokenize(reference.lower())] #reference answer tokens


    #calculate scores
    rouge_metrics = rouge_scores.score(reference, predicted)
    bleu_score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing)

    return{
        "rouge1_precision": rouge_metrics["rouge1"].precision,
        "rouge1_recall": rouge_metrics["rouge1"].recall,
        "rouge1_f1": rouge_metrics["rouge1"].fmeasure,
        "rouge2_precision": rouge_metrics["rouge2"].precision,
        "rouge2_recall": rouge_metrics["rouge2"].recall,
        "rouge2_f1": rouge_metrics["rouge2"].fmeasure,
        "rougeL_precision": rouge_metrics["rougeL"].precision,
        "rougeL_recall": rouge_metrics["rougeL"].recall,
        "rougeL_f1": rouge_metrics["rougeL"].fmeasure,
        "bleu": bleu_score
    }



# Configuration
BACKEND_URL = "http://localhost:8000"

#import requests    
#def evalution(test_cases, model_endpoint):
    #results= []

   # for test in test_cases:

        #get model output

def show():
        st.title("Model Evaluation")

        #Test case input
        with st.form("eval_form"):
              question = st.text_input("Question", "")
              reference = st.text_input("Reference Answer", "")

              if st.form_submit_button("Evaluate"):
                    try:
                        #connect to endpoint to get answer
                        response = requests.post(
                              f"{BACKEND_URL}/query",
                              json={"query": question},
                              timeout=600).json()
                                 
                        #evaluate  the output
                        metrics = evaluate_output(response["response"],reference)

                        #Display results
                        st.subheader("Results")

                        #Rouge-1
                        st.write('ROUGE-1')
                        col1,col2,col3 = st.columns(3)  
                        col1.metric("Precision",f"{metrics['rouge1_precision']:.3f}")
                        col2.metric("Recall",f"{metrics['rouge1_recall']:.3f}")
                        col3.metric("F1-Measure",f"{metrics['rouge1_f1']:.3f}")
                    
                        #Rouge-2
                        st.write('ROUGE-2')
                        col1,col2,col3 = st.columns(3)  
                        col1.metric("Precision",f"{metrics['rouge2_precision']:.3f}")
                        col2.metric("Recall",f"{metrics['rouge2_recall']:.3f}")
                        col3.metric("F1-Measure",f"{metrics['rouge2_f1']:.3f}")

                        #RougeL
                        st.write('**ROUGEL**')
                        col1,col2,col3 = st.columns(3)  
                        col1.metric("Precision",f"{metrics['rougeL_precision']:.3f}")
                        col2.metric("Recall",f"{metrics['rougeL_recall']:.3f}")
                        col3.metric("F1-Measure",f"{metrics['rougeL_f1']:.3f}")

                        #Bleu
                        st.write('Bleu Score')
                        st.metric('Score',f"{metrics['bleu']:.3f}")

                        with st.expander("Detailed Result"):
                              st.json({
                                    "Question":question,
                                    "Model Response": response["response"],
                                    "Reference": reference,
                                    "Metrics" : metrics
                                    })  
                    except Exception as e:
                          st.error(f"Evaluation failed: {str(e)}")   

if __name__ == "__main__":
      show()
        