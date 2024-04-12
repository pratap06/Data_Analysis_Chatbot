import streamlit as st
import os
from groq import Groq
import random
from pandasai import SmartDataframe
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os 
import pandas as pd
from pandasai.responses.streamlit_response import StreamlitResponse
from pandasai import Agent
from pandasai import SmartDataframe
import pickle
from langchain_community.llms import Cohere

# Load environment variables
load_dotenv()

# Load your model just once, not inside a function, to avoid reloading on each interaction
insurance_model = pickle.load(open('classifier_model.pkl', 'rb'))
model_columns = pickle.load(open('model_columns.pkl', 'rb'))

groq_api_key = os.getenv('GROQ_API_KEY')
cohere_api_key = os.getenv('COHERE_API_KEY')

def predict_insurance(input_df):
    input_df_aligned = input_df.reindex(columns=model_columns, fill_value=0)
    predictions = insurance_model.predict(input_df_aligned)
    return predictions

def chat_with_csv(df, prompt, llm_choice, model=None):
    
    if llm_choice == 'Groq':
        llm = ChatGroq(
            groq_api_key=groq_api_key, 
                model_name=model
        )
    elif llm_choice == 'Cohere':
        llm = Cohere()
    
    data = Agent(df, config={"llm": llm, "verbose": True})
    result = data.chat(prompt)
    
    return result

def main():
    st.set_page_config(layout='wide')
    st.sidebar.title('Choose your action')
    activity = st.sidebar.radio("Select Activity:", ['Insurance Claim Prediction', 'Data Analysis Chat'])

    if activity == 'Insurance Claim Prediction':
        st.title("Insurance Claim Prediction")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
            with st.spinner(text="In progress..."):
                input_df_processed = pd.get_dummies(input_df)
         # Align the dataframe with the model columns, filling missing columns with 0s
                input_df_aligned = input_df_processed.reindex(columns=model_columns, fill_value=0)
            # Perform prediction
                predictions = insurance_model.predict(input_df_aligned)
            # Adding the predictions to the dataframe for display
                input_df['Predictions is_claim'] = predictions
            if st.button("Predict"):
                    with st.spinner(text="Predictions are In progress..."):
                # Display the dataframe with predictions
                        st.write("Next 6 months Predictions:")
                        st.dataframe(input_df)
        else:
            st.write("Please upload a CSV file to start the prediction.")

    elif activity == 'Data Analysis Chat':
        st.title("Data Analysis Chat App")
        input_csv = st.file_uploader("Upload your CSV file", type=['csv'])
        
        # LLM selection moved to a new sidebar section
        st.sidebar.title('LLM Options')
        llm_choice = st.sidebar.radio(
            "Choose an LLM for analysis:",
            ['Groq', 'Cohere']
        )

        if llm_choice == 'Groq':
            model = st.sidebar.selectbox(
                'Choose a model',
                ['mixtral-8x7b-32768', 'llama2-70b-4096','gemma-7b-it']
            )
        else:
            model = None
        

        
        
            
        if input_csv is not None:

            col1, col2 = st.columns([1,1])

            with col1:
                st.info("CSV Uploaded Successfully")
                df = pd.read_csv(input_csv)
                st.dataframe(df, use_container_width=True)

            with col2:

                st.info("Chat Below")
            
                input_text = st.text_area("Enter your query")

                if input_text is not None:
                    if st.button("Check Results"):
                        with st.spinner(text="In progress..."):
                            st.info("Your Query: "+input_text)
                            result = chat_with_csv(df, input_text,llm_choice,model)
                            st.success(result)

if __name__ == "__main__":
    main()