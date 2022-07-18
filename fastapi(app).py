#making an API for sentiment analysis 
#firstly defining all the important function and importing libraries
import nltk
import re
import pickle
import string
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

def preprocess_text(text):
  #first lowercasing all the letters 
  text=text.lower()
  #striiping our text 
  text = text.strip()
  #reoving any HTML markups 
  text=re.compile('<.*?>').sub('', text) 
  text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)  
  text = re.sub('\s+', ' ', text)  #Remove extra space and tabs
  text = re.sub(r'\[[0-9]*\]',' ',text) #[0-9] matches any digit (0 to 10000...)
  text=re.sub(r'[^\w\s]', '', str(text).lower().strip())
  text = re.sub(r'\d',' ',text) #matches any digit from 0 to 100000..., \D matches non-digits
  text = re.sub(r'\s+',' ',text) #\s matches any whitespace, \s+ matches multiple whitespace, \S matches non-whitespace 

  return text


def stopword(string):
    a= [i for i in string.split() if i not in stopwords.words('english')]
    return ' '.join(a)

snow = SnowballStemmer('english')
def stemming(string):
    a=[snow.stem(i) for i in word_tokenize(string) ]
    return " ".join(a)


wl = WordNetLemmatizer()
 
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def lemmatizer(string):
    word_pos_tags = nltk.pos_tag(word_tokenize(string)) # Get position tags
    a=[wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token
    return " ".join(a)

def finalpreprocess(string):
    return lemmatizer(stopword(preprocess_text(string)))



#loading the model 
with open('C:\codes\machine learning for complete beginners\True foundry\lnb_tfidf.pkl' , 'rb') as f:
    lr_tfidf = pickle.load(f)

with open('C:\codes\machine learning for complete beginners\True foundry\lfidf_vectorizer.pickle' , 'rb') as g:
    tfidf = pickle.load(g)


#maing the APP for it 

from fastapi import FastAPI
import uvicorn
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from pydantic import BaseModel

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
    text=finalpreprocess(test_data)
    text=[text]
    text_vector=tfidf.transform(text)
    y_predict = lr_tfidf.predict(text_vector)
    # Predicting the Class
     
    # Return the Result
    return {'prediction':y_predict}
if __name__ == '__main__':

    uvicorn.run(app, host='127.0.0.1', port=4000, debug=True)