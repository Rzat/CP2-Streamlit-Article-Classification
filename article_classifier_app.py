#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 20:22:45 2024

@author: rajatthakur
"""
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
import pandas as pd

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

st.set_page_config(layout="wide", page_title="Article Classification App")
st.sidebar.title("About This App")
st.sidebar.write("This app classifies articles into categories using a trained model.")



tfidf = joblib.load('tfidf_vectorizer.pkl')
nb_model = joblib.load('nb_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

def predicted_label(article,model):
    processed_text = re.sub('[^a-zA-Z]', ' ', article).lower()
    stop_words = set(stopwords.words('english'))
    processed_text = ' '.join([word for word in processed_text.split() if word not in stop_words])
    text_tfidf = tfidf.transform([processed_text]).toarray()
    probabilities = model.predict_proba(text_tfidf)[0]
    predicted_class = model.predict(text_tfidf)[0]
    confidence = probabilities[predicted_class]
    return predicted_class, confidence, probabilities
    


def main():
    st.title("Article Classification App")
    st.write("Welcome to article classification app. Enter an article below to classify it")

    col1, col2 = st.columns([2,2])
    
   
    with col1:
   
     article_input= st.text_area("Input article text", height=200)
 
    if st.button("Classify"):
            
     if article_input.strip():
            st.write("Processing your Article...") 
            numeric_label, confidence, probabilities = predicted_label(article_input, nb_model)
            readable_label = label_encoder.inverse_transform([numeric_label])[0]
            st.success(f"The predicted category is: **{readable_label}** with **{confidence * 100:.2f}%** Confidence")
        
            with col2:
                st.write("Class Probabilities: ")
                class_labels = label_encoder.classes_
                st.bar_chart(pd.DataFrame(probabilities, index=class_labels, columns=["Probability"]))
                st.write(f"Confidence: {confidence:.2%}")
                st.progress(confidence)
                # result_df = pd.DataFrame({
                #     "Class": label_encoder.classes_,
                #     "Probability": probabilities[0]
                #     })
                # csv = result_df.to_csv(index=False)
                # st.download_button("Download Report as CSV", data=csv, file_name="classification_report.csv", mime="text/csv")
                
                
    else:
            st.warning("Please enter some text to classify")

            
    


if __name__ == "__main__":
    main()
