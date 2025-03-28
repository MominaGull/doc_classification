import numpy as np
import spacy
import nltk
import tensorflow as tf
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize


def streamlit_config():

    # page configuration
    st.set_page_config(page_title='Document Classification', layout='centered')

    # Sidebar for model selection
    with st.sidebar:
        st.header("Model Selection")
        model_option = st.selectbox(
            "Choose a model",
            ["Bidirectional LSTM", "gpt-4"],
            index=0
        )
        
        if st.button("Select Model"):
            st.session_state.model = model_option
            st.success(f"Selected model: {model_option}")
    
    # page header transparent color
    page_background_color = """
    <style>

    [data-testid="stHeader"] 
    {
    background: rgba(0,0,0,0);
    }

    </style>
    """
    st.markdown(page_background_color, unsafe_allow_html=True)

    # title and position
    st.markdown(f'<h1 style="text-align: center;">Financial Document Classification</h1>',
                unsafe_allow_html=True)
    add_vertical_space(4)


def extract_data_from_html(html_file):
    '''
    Extract data from the html file
    :param html_file: the path to the html file
    :return: a list of words
    '''
    content = html_file.read().decode('utf-8')
    
    soup = BeautifulSoup(content, 'html.parser')
    all_text = soup.get_text()
    result = [text.strip() for text in all_text.split()]
    return result


def preprocessing(text):
    '''
    Preprocess the text data
    :param text: the text data
    :return: the preprocessed text data
    '''
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(' '.join(text))
    tokens_list = [token.lemma_.lower().strip()
                   for token in doc if token.text.lower() not in nlp.Defaults.stop_words and token.text.isalpha()
    ]

    if tokens_list:
        return ' '.join(tokens_list)
    else:
        return "No tokens found"
    

def sentence_embeddings(sentence):
    '''
    Create sentence embeddings using the word2vec model
    :param sentence: the sentence
    :param model: the word2vec model
    :return: the sentence embeddings
    '''
    words = word_tokenize(sentence)
    model = Word2Vec.load('word2vec_model.bin')
    vectors = [model.wv[word] for word in words if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)


def prediction(input_file):

    html_content = extract_data_from_html(input_file)
    preprocessed_text = preprocessing(html_content)
    features = sentence_embeddings(preprocessed_text)

    features = np.expand_dims(features, axis=0)
    features = np.expand_dims(features, axis=2)

    features_tensors = tf.convert_to_tensor(features, dtype=tf.float32)

    if st.session_state.model == "Bidirectional LSTM":
        model = tf.keras.models.load_model('model.keras', custom_objects = {'Orthogonal': tf.keras.initializers.Orthogonal})
    prediction = model.predict(features_tensors)

    target_label = np.argmax(prediction)

    target = {0: "Balance Sheets", 1: "Income Statement",
              2: "Cash Flow", 3: "Notes", 4: "Others"}
    predicted_class = target[target_label]

    confidence = round(np.max(prediction)*100, 2)

    add_vertical_space(2)
    st.markdown(f'<h3 style="text-align: center; color: white;">Predicted Label = {predicted_class}</h3>', 
                    unsafe_allow_html=True)
    
    add_vertical_space(1)
    st.markdown(f'<h3 style="text-align: center; color: white;">With {confidence}% Confidence</h4>', 
                    unsafe_allow_html=True)

streamlit_config()
    

# File Uploader
input_file = st.file_uploader('Upload an HTML file', type='html')

# check if the file is not None
if input_file is not None:

    # predict the class of the document
    prediction(input_file)

    # except:
    #     try:
    #         nltk.data.find('tokenizers/punkt')
    #     except LookupError:
    #         nltk.download('punkt')

    #     prediction(input_file)