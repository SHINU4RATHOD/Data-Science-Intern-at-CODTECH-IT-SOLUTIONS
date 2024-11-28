import pandas as pd
import seaborn as sns
################# loading data1
imbalance_data = pd.read_csv(r"C:/Users/SHINU RATHOD/Desktop/internship assignment/08_CodeTech soln/Data Scientist/05_NLP_Application/02_Task_NLP_hate_speech_rec/imbalanced_data.csv")     # loading dataset
imbalance_data.head()
imbalance_data.info()
imbalance_data.shape
imbalance_data.isnull().sum()   # we can see that there no na/missing value available in the dataset

imbalance_data.drop("id", axis=1, inplace=True)

sns.countplot(x='label', data=imbalance_data)


################## loading raw dataset
raw_data = pd.read_csv(r"C:/Users/SHINU RATHOD/Desktop/internship assignment/08_CodeTech soln/Data Scientist/05_NLP_Application/02_Task_NLP_hate_speech_rec/raw_data.csv")   # reading second dataset
raw_data.head()
raw_data.info()
imbalance_data.shape
raw_data.isnull().sum()

# Let's drop the columns which are not required for us.
raw_data.drop(['Unnamed: 0','count','hate_speech','offensive_language','neither'],axis=1,inplace =True)
raw_data.info()

sns.countplot(x='class',data =raw_data)

raw_data['class'].unique()  # Let's check for the unique values in the dataset
#copying the valus of the class 1 into class 0.
raw_data[raw_data['class']==0]['class']=1
# Let's check the values in the claass 0
raw_data[raw_data['class']==0]
# replace the value of 0 to 1
raw_data["class"].replace({0:1},inplace=True)
raw_data["class"].unique()  

sns.countplot(x="class",data= raw_data)

# Let's replace the value of 2 to 0.
raw_data["class"].replace({2:0}, inplace = True)

sns.countplot(x='class',data=raw_data)

imbalance_data.head()
raw_data.head()

# Let's change the name of the 'class' to label
raw_data.rename(columns={'class':'label'},inplace =True)
raw_data.head()

################################## concatinate both the data into a single data frame.
frame = [imbalance_data, raw_data]
df = pd.concat(frame)
df.info()

sns.countplot(x='label',data=df)



# Data preprocessing
import re
import nltk
import string
from nltk.corpus import stopwords
nltk.download('stopwords')
# Let's apply stemming and stopwords on the data
stemmer = nltk.SnowballStemmer("english")
stopword = set(stopwords.words('english'))

# defining function for to clean dataset
def data_cleaning(words):
    words = str(words).lower()
    words = re.sub('\[.*?\]', '', words)
    words = re.sub('https?://\S+|www\.\S+', '', words)
    words = re.sub('<.*?>+', '', words)
    words = re.sub('[%s]' % re.escape(string.punctuation), '', words)
    words = re.sub('\n', '', words)
    words = re.sub('\w*\d\w*', '', words)
    words = [word for word in words.split(' ') if words not in stopword]
    words=" ".join(words)
    words = [stemmer.stem(words) for word in words.split(' ')]
    words=" ".join(words)
    return words

df["tweet"][1]
# let's apply the data_cleaning on the data.
df['tweet']=df['tweet'].apply(data_cleaning)
df["tweet"][1]

########## extracting independent and dependent var
x = df['tweet']
y = df['label']

##################### splitting the dataset into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, random_state = 42)
x_train.shape, x_test.shape, y_train.shape, y_test.shape

########################### feature engineering
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

max_words = 50000
max_len = 300

tokenizer = Tokenizer(num_words=max_words)  # Limits the vocabulary size to the max_words most frequent words in the training data. Words outside this range will be ignored.
tokenizer.fit_on_texts(x_train)    #Analyzes the texts in x_train (training data) to create:
# A word index (mapping each word to a unique integer based on frequency)

sequences = tokenizer.texts_to_sequences(x_train) #Converts each text in x_train into a sequence of integers. Each word is replaced by its corresponding index from the tokenizer’s vocabulary.
sequences_matrix = pad_sequences(sequences,maxlen=max_len) #Ensures that all sequences are the same length (max_len)
sequences_matrix

########################### model building
from keras.models import Sequential
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, SpatialDropout1D
from keras.optimizers import RMSprop
# Creating model architecture.
model = Sequential()
model.add(Embedding(max_words,100,input_length=max_len))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100,dropout=0.2,recurrent_dropout=0.2))
model.add(Dense(1,activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
history = model.fit(sequences_matrix,y_train,batch_size=128,epochs = 5,validation_split=0.2)


test_sequences = tokenizer.texts_to_sequences(x_test)
test_sequences_matrix = pad_sequences(test_sequences,maxlen=max_len)
test_sequences_matrix

# Model evaluation
accr = model.evaluate(test_sequences_matrix,y_test)
lstm_prediction = model.predict(test_sequences_matrix)

res = []
for prediction in lstm_prediction:
    if prediction[0] < 0.5:
        res.append(0)
    else:
        res.append(1)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,res))

import pickle
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
# saving the mdoel.
model.save("model.h5")
# loading the saved model
import keras        
load_model=keras.models.load_model("model.h5")
with open('tokenizer.pickle', 'rb') as handle:
    load_tokenizer = pickle.load(handle)
    
    
    
    
    
    
    
    
##############################################test our model on custom data.
test = 'i love this movie'

def clean_text(text):
    print(text)
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    print(text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text

test=[clean_text(test)]
print(test)

seq = load_tokenizer.texts_to_sequences(test)
padded = pad_sequences(seq, maxlen=300)
print(seq)

pred = load_model.predict(padded)

print("pred", pred)
if pred<0.5:
    print("no hate")
else:
    print("hate and abusive")
    

