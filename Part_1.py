#!/usr/bin/env python
# coding: utf-8

# # Data Mining: Classification analysis on textual data

# ## Idea

# Our brains can easily perceive the meaning of the word in the context it is used. How will a machine learning algorithm do the same? This idea is quite interesting, ain't it? Suppose it's a library and we ask librarian to sort out 10,000 books by its genre. It would take a couple of days to do so. What if we have 10,000 digital books and we ask our algorithm to do the sorting? Is it possible? Yes, it is. 

# ## Data and what it represents

# I've used 20 newsgroups dataset for classification analysis of textual data. You can download the dataset from [here.](https://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups)

# ## Dependencies

# This project has been coded using Python 3.7.1 under environment Anaconda Jupyter Notebook.
# Please install the following packages before running the code.
# #### 1.  nltk v3.2.2
# #### 2. numpy v1.11.3
# #### 3. matplotlib v2.0.0
# #### 4. sklearn v0.18.1

# # Part 1: Model Text Data and Feature Extraction

# The algorithm can't read textual data. So, textual data needs to be encoded as integers or floating point values. 
# Let's go through step-by-step. 
# 
# Step 1: Tokenization: Parsing text to remove words. 
# 
# Step 2: Feature extraction (or vectorization): Encoding words as integers to be fed as input to algorithm.
# 
# Read more about how to do this [here](https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/)
# 
# Read about 'Bag of words' and 'TFIDF' method. This basically helps us converting textual data into a vectorised array of floating values. 

# ### Let's import dataset from scikit library

# In[3]:


from sklearn.datasets import fetch_20newsgroups


# There are total of 20 classes. You can find the [list of classes here](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html) 
# 
# For the sake of understanding, we will consider only the following four classes: 
# 
#     'comp.graphics',
# 
#     'comp.os.ms-windows.misc',
#  
#     'comp.sys.ibm.pc.hardware',
#  
#     'comp.sys.mac.hardware'.
#  
# **Since this is about computer technology, we will henceforth consider only one class: 'Computer Technology'. The abovementioned will be four subclasses.**
# 
# Let's then make a list of these subclasses as follows.

# In[4]:


computer_technology_subclasses=['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware']


# In[5]:


computer_technology_subclasses


# # Preprocessing data
# 
# We need to make train and test data for the above class. This is quite easy to make.
# If you look at docstring of fetch_20newsgroups under, you can set subset to either 'train'/'test'/'all'. 

# In[6]:


help(fetch_20newsgroups)


# **We also need to remove headers, footers, quotes, punctuations, stop words(eg., the, and, or). Stemming is used to remove suffixes of similar words.**

# In[7]:


# Forming train and test data
computer_train=fetch_20newsgroups(subset='train',categories=computer_technology_subclasses,shuffle=True,random_state=42,
                                  remove=('headers','footers','quotes'))
computer_test=fetch_20newsgroups(subset='test',categories=computer_technology_subclasses,shuffle=True,random_state=42,
                                 remove=('headers','footers','quotes'))


# In[8]:


from nltk.stem.snowball import SnowballStemmer


# In[9]:


#defining the stemmer to be used in preprocessing the data
stemmer=SnowballStemmer("english")
stemmer


# In[10]:


#defining the list of punctutations to be trimmed off the data in the preprocessing stage
punctuations='[! \" # $ % \& \' \( \) \ * + , \- \. \/ : ; <=> ? @ \[ \\ \] ^ _ ` { \| } ~]'
punctuations


# In[11]:


import re 
#You can find more information by using Shift+Tab on re


# In[12]:


#function for stemming, and removing punctuations
def preprocess(data):
    for i in range(len(data)):
        data[i]=" ".join([stemmer.stem(data) for data in re.split(punctuations,data[i])])
        data[i]=data[i].replace('\n','').replace('\t','').replace('\r','')


# In[13]:


#preprocess the two datasets
preprocess(computer_train.data)
preprocess(computer_test.data)


# In[14]:


type(computer_train.data)


# ### CountVectorizer and TfidfVectorizer

# The **CountVectorizer** provides a simple way to both tokenize a collection of text documents and build a vocabulary of known words, but also to encode new documents using that vocabulary.
# 
# The **TfidfVectorizer** will tokenize documents, learn the vocabulary and inverse document frequency weightings, and allow you to encode new documents. Alternately, if you already have a learned **CountVectorizer**, you can use it with a **TfidfTransformer** to just calculate the inverse document frequencies and start encoding documents.

# In[15]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


# Let's create our tfidf matrix. 
# 
# CountVectorizer will create the vocabulary of known words.

# In[16]:


#Creating the instance of CountVectorizer class and removing stop_words.
#min_df = 2 means that if any word occurs rarely or has frequency of occurence lower than 2, it will be considered irrelevant
# and discarded.
vectorizer=CountVectorizer(min_df=2,stop_words ='english')
vectorizer


# In[17]:


#tokenize and build vocab
vectorizer_counts=vectorizer.fit_transform(computer_train.data+computer_test.data)
vectorizer_counts


# Sparse matrix means that the matrix will have a lot of zeros in it.

# In[44]:


#Summarize
#print(vectorizer.vocabulary_)
#Getting feature names
vectorizer.vocabulary_.keys()
vocab = list()
for i in vectorizer.vocabulary_.keys():
    vocab.append(i)

vocab


# This creates a dictionary of words and indices. Each word is a key and a number is assigned to each key as index.

# In[19]:


import pandas as pd
pd.DataFrame(vectorizer.vocabulary_, index = [0])


# In[20]:


#Encode the document
tfidf_transformer=TfidfTransformer()
tfidf_transformer


# In[21]:


vectorizer_tfidf = tfidf_transformer.fit_transform(vectorizer_counts)
vectorizer_tfidf


# So, we have converted the documents of 4 classes into numerical feature vectors by first tokenising each document into words and then excluding stop words, punctuations. 
# 
# Then after, we created TFxIDF vector representations. 
# 
# 
# We will conclude by reporting the number of terms we extracted.
# 

# In[22]:


print('Min Frequency: 2')
print('Number of Terms: '+str(vectorizer_tfidf.shape[1]))


# In[51]:


#Summarize encoded vector
print(vectorizer_tfidf.shape)


# In[24]:


print(vectorizer_tfidf.toarray())


# In[32]:


import pandas as pd


# In[30]:


#pd.DataFrame(vectorizer_tfidf.toarray())


# In[27]:


#pd.DataFrame(vectorizer_tfidf.data)


# ## Finding 10 most significant terms

# Before, we found out most important terms/words in a document using TFIDF metric. We will use the same concept for finding out the most important term in a class for each subset: 'All', 'Train', 'Test'.
# * Class: 
#     * comp.sys.ibm.pc.hardware
#     * comp.sys.mac.hardware
#     
# We have already install all the required libraries. 

# In[28]:


#Defining dictionary of subsets
pretty_print={'all': 'All subsets included','train': 'Only training subsets included','test': 'Only testing subsets included'}


# In[48]:


def ten_most_significant_helper_min_df_2(newsgroup,subset):
    data=fetch_20newsgroups(subset=subset,categories=[newsgroup],shuffle=True,random_state=42,remove=('headers','footers','quotes')).data
    punctuations='[! \" # $ % \& \' \( \) \* + , \- \. \/ : ; <=> ? @ \[ \\ \] ^ _ ` { \| } ~]'

    #preprocessing using the stemmer, and stripping off the punctuations
    for i in range(len(data)):
        data[i]=" ".join([stemmer.stem(data) for data in re.split(punctuations,data[i])])
        data[i]=data[i].replace('\n','').replace('\t','').replace('\r','')
    
    #top 10 features required; also removing stopwords is necessary
    counts=CountVectorizer(min_df=2,max_features=10,stop_words ='english')
    X_counts=counts.fit_transform(data)
    print('\n')
    print('Class being operated upon:'+newsgroup)
    print('The data subset:'+pretty_print[subset])
    print('10 most significant terms with min_df=2:')
    
    #Getting list of feature names
    feature_names = list()
    for i in counts.vocabulary_.keys():
        feature_names.append(i)
        
    
    for (term,count) in zip(feature_names,X_counts.toarray().sum(axis=0)):
        spaces=''
        for i in range(15):
            if 15-i-len(term) > 0:
                spaces += ' '
        print(spaces+'\"'+term+'='+str(count))
        
    tfidf_transformer=TfidfTransformer()
    class_tficf=tfidf_transformer.fit_transform(X_counts)
    class_tficf_arr = class_tficf.toarray()
    print('TFxICF dimension:',)
    print(class_tficf.shape)


# If you use Spyder as your environment, you can easily get feature names using get_feature_names() method on CountVectorizer instance. But you can't do that in jupyter.
# 

# In[49]:


#helpers for the 10 most significant terms in the total, train, and test subsets of the dataset    
def ten_most_significant_min_df_2(newsgroup):
    ten_most_significant_helper_min_df_2(newsgroup,'all')
    ten_most_significant_helper_min_df_2(newsgroup,'train')
    ten_most_significant_helper_min_df_2(newsgroup,'test')


# In[50]:


ten_most_significant_min_df_2('comp.sys.ibm.pc.hardware')
ten_most_significant_min_df_2('comp.sys.mac.hardware')


# In[ ]:




