
# coding: utf-8

# # Acronyms in the data and replacing
# 
# ##### Get all acronyms in data and replace them with meaningful words

# In[17]:


import nltk
import spacy
import pandas as pd
import numpy as np

import re
from collections import Counter
from itertools import chain
from nltk.corpus import stopwords
from textblob import TextBlob

pd.options.display.max_columns = None


# In[2]:


#load data
data = pd.read_csv('/home/haomiao/work/target0/data/initial_cleaned_text_Airsweb.csv')


# In[3]:


#get the columns that contains acronyms to list
raw_remedial = data['Immediate remedial actions taken'].tolist()
raw_remedial.extend(data['What were you able to do about it?'].tolist())
raw_follow_up = data['Description of follow-up'].tolist()
raw_incident = data['Describe the incident:'].tolist()
raw_incident.extend(data['What did you see?'].tolist())
raw_incident.extend(data['What could have happened?'].tolist())
#remove nan
cleaned_raw_remedial = [x for x in raw_remedial if str(x) != 'nan']
cleaned_raw_follow_up = [x for x in raw_follow_up if str(x) != 'nan']
cleaned_raw_incident = [x for x in raw_incident if str(x) != 'nan']


# In[5]:


acronyms_list = {'s&c':'S&C',
                 's&t':'S&T',
                 'h&s':'H&S',
                 'L&m':'L&M',
                 'H&s':'H&S',
                 'e&p':'E&P',
                 'h&S':'H&S'}


# In[12]:


#replace lowercase into uppercase then replace the word 
def multipleReplace(text, wordDict=acronyms_list):
    for key in wordDict:
        text = text.replace(key, wordDict[key])
    return text
cleaned_raw_follow_up= [multipleReplace(text) for text in cleaned_raw_follow_up]
cleaned_raw_incident= [multipleReplace(text) for text in cleaned_raw_incident]
cleaned_raw_remedial= [multipleReplace(text) for text in cleaned_raw_remedial]


# In[15]:


#get acronyms and count
acronyms = []
datalist = [cleaned_raw_follow_up, cleaned_raw_incident, cleaned_raw_remedial]
#pattern = r"[A-Z][a-zA-Z]*[A-Z]|[a-zA-Z]*\&[a-zA-Z]|(?:[A-Z]\.)+"
pattern = r"[a-zA-Z]*\&[a-zA-Z]*"
for data in datalist:
    for text in data:
        acronyms.extend(re.findall(pattern,text))
acronyms_count = pd.Series(Counter(acronyms)).sort_values(ascending = False)





#%%
#get tag, lemma, replace tag for incidents data

nlp = spacy.load('en')
doc = map(nlp,cleaned_raw_incident)
tokens_raw = [[(token.text, token.lemma_, token.tag_) for token in sent] for sent in doc]


words = list(chain(*tokens_raw))


# In[54]:


verbs = {'VB',
         'VBD',
         'VBG',
         'VBN',
         'VBP',
         'VBZ' }
nouns = {'NN',
         'NNS',
         'NNP',
         'NNPS'}


# In[55]:


def retag_word(word):
    if word[2] in verbs:
        return (word[0],word[1],'V')
    elif word[2] in nouns:
        return (word[0],word[1],'N')
    else:
        return word
    
def retag_sentence(sentence):
    return [retag_word(word) for word in sentence]


# In[56]:


retagged_words = [retag_word(word) for word in words]


#%%


#remove punct
punct=[",",".",":","(",")","-","/", " ","*","#"]
retagged_words = [i for i in retagged_words if i[0] not in punct ]
#remove stop words
filtered_words = [word for word in retagged_words if word[0] not in stopwords.words('english')]
#lowercase everything
filtered_words = [(word[0].lower(),word[1],word[2]) for word in filtered_words]


# In[82]:


verbs_nouns_incident = [i for i in filtered_words if i[2] in {"V","N"}]


# In[83]:

v_n_incident_df = pd.Series(Counter(word for word in verbs_nouns_incident)).sort_values(ascending = False).reset_index()
v_n_incident_df.rename(columns ={'level_0':'word','level_1':'word_lemma','level_2':'tag',0:'count'},inplace = True)

#%%
verbs_incident = v_n_incident_df.loc[v_n_incident_df['tag'] == 'V']
nouns_incident = v_n_incident_df.loc[v_n_incident_df['tag'] == 'N']

#%%
verbs_incident.sort_values(['count'],ascending =False,inplace = True)
nouns_incident.sort_values(['count'],ascending =False,inplace = True)
#%%
verbs_incident.to_csv('/home/haomiao/scratch/Target0/verbs_incident.csv')
nouns_incident.to_csv('/home/haomiao/scratch/Target0/nouns_incident.csv')
print(len(verbs_incident['word_lemma'].unique()))
print(len(nouns_incident['word_lemma'].unique()))
