import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import pandas as pd


# @st.cache(allow_output_mutation=True)
@st.experimental_singleton
def load_model():

    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

    return model, tokenizer



def predict_ner(input_text):
    ner_results = nlp(input_text)
    entity_type = []
    word = []

    for i in range(len(ner_results)):
        entity_type.append(ner_results[i]['entity'][2:])
        word.append(ner_results[i]['word'])

    df = pd.DataFrame(list(zip(word, entity_type)),
               columns =['Word', 'Entity_type'])
    return df



st.title("Named Entity Recognition")

input_text = st.text_input("Input Text","Type Here")
if st.button("Predict"):
    model, tokenizer = load_model()
    nlp = pipeline("ner", model=model, tokenizer=tokenizer) 
    #res = predict_ner(input_text)
    ner_results = nlp(input_text)
    entity_type = []
    word = []

    for i in range(len(ner_results)):
        entity_type.append(ner_results[i]['entity'][2:])
        word.append(ner_results[i]['word'])

    res = pd.DataFrame(list(zip(word, entity_type)),
               columns =['Word', 'Entity_type'])
    st.success('The output:::')

    hide_dataframe_row_index = """
            <style>
            .row_heading.level0 {display:none}
            .blank {display:none}
            </style>
            """
    st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
    st.table(res)

# if __name__=='__main__':
#     main()
