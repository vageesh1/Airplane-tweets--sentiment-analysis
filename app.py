import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import load_model

#loading our model 
model=load_model('C:\codes\machine learning for complete beginners\True foundry\LSTM_model.h5')

#now doing our predictions 
from fastapi import FastAPI
import uvicorn
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from pydantic import BaseModel

max_words=1000
def preprocessing(text):
    tok1 = Tokenizer(num_words=max_words)
    tok1.fit_on_texts(text)
    sequences_train = tok1.texts_to_sequences(text)
    sequences_train_matrix = sequence.pad_sequences(sequences_train,maxlen=150)
    return sequences_train_matrix

data={'text':'plus youve added commercials to the experience... tacky',}
 
# Creating FastAPI instance
app = FastAPI()
 
# Creating class to define the request body
# and the type hints of each attribute
class request_body(BaseModel):
    text: str #our input format will be string

 
#creating an endpoint for the request
@app.post('/predict')

def predict(data : request_body):
    data=data.dict()
    # Making the data in a form suitable for prediction
    test_data = data['text']
    
    
    # Preprocessing the text 
    text=preprocessing(test_data)
    prediction=model.predict(text)
    # Predicting the Class
     
    # Return the Result
    return {'prediction':prediction.tolist()}
if __name__ == '__main__':

    uvicorn.run(app, host='127.0.0.1', port=4000, debug=True)