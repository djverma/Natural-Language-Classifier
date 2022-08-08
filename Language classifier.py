#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import re
from nltk.tokenize import sent_tokenize
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
import os
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pandas as pd


# In[ ]:



pattern = r'<(!?).*>'
MAX_LENGTH_TRANSCRIPTION = 150000


# In[ ]:



labels = { 
          1.0:'Danish', 2.0:'German',
          3.0: 'Greek', 4.0: 'English', 
          5.0: 'Spanish',6.0: 'Finnish', 
          7.0: 'French', 8.0: 'Italian', 
          9.0: 'Dutch', 10.0: 'Portuguese', 
          11.0: 'Swedish', 12.0: 'Bulgarian',
          13.0: 'Czech', 14.0: 'Estonian',
          15.0: 'Hungarian', 16.0: 'Lithuanian',
          17.0: 'Latvian', 18.0: 'Polish',
          19.0: 'Romanian', 20.0: 'Slovak',
          21.0: 'Slovenian'
        
language_codes_files = {
    'Danish': ['da', '/ep-00-01-17.txt'], 'German': ['de', '/ep-00-01-17.txt'], 
    'Greek': ['el', '/ep-00-01-17.txt'], 'English': ['en', '/ep-00-01-17.txt'], 
    'Spanish': ['es', '/ep-00-01-17.txt'], 'Finnish': ['fi', '/ep-00-01-17.txt'],
    'French': ['fr','/ep-00-01-17.txt'], 'Italian': ['it', '/ep-00-01-17.txt'], 
    'Dutch': ['nl', '/ep-00-01-17.txt'], 'Portuguese': ['pt', '/ep-00-01-17.txt'], 
    'Swedish': ['sv', '/ep-00-01-17.txt'], 'Bulgarian': ['bg', '/Bulgarian.txt'],
    'Czech': ['cs', '/Czech.txt'], 'Estonian': ['et', '/Estonian.txt'],
    'Hungarian': ['hu', '/Hungarian.txt'], 'Lithuanian': ['lt', '/Lithuanian.txt'],
    'Latvian': ['lv', '/Latvian.txt'], 'Polish': ['pl', '/Polish.txt'],
    'Romanian': ['ro', '/Romanian.txt'], 'Slovak': ['sk', '/Slovak.txt'],
    'Slovenian': ['sl', '/Slovenian.txt']
}

limited_raw_text = ['Bulgarian', 'Czech', 'Estonian', 'Hungarian', 'Lithuanian',
                     'Latvian', 'Polish', 'Romanian', 'Slovak', 'Slovenian']


# In[ ]:


def combine_text_files(language_code, language):
    
    file_name_list = os.listdir('language_data/txt/' + language_code + '/')
    language_transcription = ''
    for file_name in file_name_list:
        if(len(language_transcription) >= MAX_LENGTH_TRANSCRIPTION):
            break;
        path = os.getcwd() + '/language_data/txt/' + language_code + '/' + file_name
        with open(path) as f:
            contents = f.read()
            language_transcription += contents
    
    write_path = os.getcwd() + '/language_data/txt/' + language_code + '/' + language + '.txt'
    with open(write_path, 'w') as f:
        f.write(language_transcription)

def read_languages_data(path):
    with open(path) as f:
        language_transcription = f.read()
        language_transcription = language_transcription[:MAX_LENGTH_TRANSCRIPTION]
    return language_transcription

def clean_sentences(sentences):
    for i, sentence in enumerate(sentences):
        sentences[i] = re.sub(pattern,'',sentence)

def combine_language_data(sentences, language_index):
    sentences = np.array(sentences)
    sentences = sentences.reshape(sentences.shape[0],1)
    target = np.zeros((sentences.shape[0],1))
    target += language_index
    language_data = np.hstack((sentences, target))
    return language_data

def test_languages(X_test, true_values, predictions):
    true_values = np.array(true_values)
    
    for i, sentence in enumerate(X_test):
        prediction = float(predictions[i])
        true_value = float(true_values[i])
        print("Prediction: " + str(labels[prediction]))
        print("Actual Language: " + str(labels[true_value]))
        print("Input Sentence: ")
        print(sentence)
        print('\n')

def shuffle_rows(languages):
    index = np.arange(0, len(languages))
    np.random.shuffle(index)
    shuffled_languages = languages[index,:]

    return shuffled_languages
    
def preproccess_raw_data(file_paths):
    language_codes_files_subset = dict( (key, language_codes_files[key] ) for key in limited_raw_text if key in language_codes_files )
    for language in language_codes_files_subset.keys():
        combine_text_files(language_codes_files_subset[language][0], language)
    language_transcriptions = [ read_languages_data(path) for path in file_paths ]

    for i, language_transcription in enumerate(language_transcriptions):
        language_transcriptions[i] = sent_tokenize(language_transcription)
    for sentences in language_transcriptions:
        clean_sentences(sentences)
    languages = [ combine_language_data(sentences,i+1) for i,sentences in enumerate(language_transcriptions) ]

    languages =  np.vstack((languages))
    languages = shuffle_rows(languages)
    
    return languages


# In[ ]:



file_paths = [ os.getcwd() + '/language_data/txt/' + language_codes_files[language][0] + language_codes_files[language][1] for language in language_codes_files ]

# Preprocess all raw text into a form suitable for TfidfVectorizer
languages = preproccess_raw_data(file_paths)
languages
df_languages = pd.DataFrame(languages)
df_languages.columns = ['natural language', 'language index']
df_languages['language index'] = df_languages['language index'].apply(float)
df_languages['language'] = df_languages['language index'].map(labels)
print(df_languages.isnull().any())
display(df_languages.head(10))
lit data into raw features and labels

language_features = df_languages['natural language']
language_targets = df_languages['language index']
unique, counts = np.unique(language_targets, return_counts=True)
dict(zip(unique, counts))
X_train, X_test, y_train, y_test = train_test_split(language_features, 
                                                    language_targets,
                                                    test_size = 0.3,
                                                    random_state = 42)


# In[ ]:


tfidf_vect = TfidfVectorizer(analyzer='char', ngram_range=(1,3))
model = MultinomialNB()
text_clf = Pipeline([('tfidf', tfidf_vect),
                    ('clf', model),
                    ])
predictions = text_clf.predict(X_test)


# In[ ]:


accuracy_score(y_test,predictions)


# In[ ]:


scores = cross_val_score(text_clf, language_features, language_targets, cv=5)
print("Mean cross-validation accuracy: " + str(scores.mean()))


# In[1]:


language_names = list(language_codes_files.keys())
plt.figure(figsize=(32, 32))
cm = confusion_matrix(y_test, predictions)

ax = sns.heatmap(cm, annot = True, fmt = "d")

ax.set_xlabel('Predicted Language')
ax.set_ylabel('Actual Language')
ax.set_title('Language Identification Confusion Matrix')
ax.set_xticklabels(labels.values())
ax.set_yticklabels(labels.values())
plt.show()
test_languages(X_test, y_test, predictions)


# In[ ]:




