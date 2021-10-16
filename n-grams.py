import numpy as np
import pandas as pd
import nltk, re, string, collections
from nltk.util import ngrams
from nltk.corpus import stopwords, shakespeare
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import gutenberg
from sklearn.feature_extraction.text import TfidfVectorizer

def remove_stopewords(word_tokens):
    stop_words = set(stopwords.words("english"))
        #print(stop_words)
    output_text = [w for w in word_tokens if not w.lower() in stop_words]
    output_text = []
    for w in word_tokens:
        if w not in stop_words:
            output_text.append(w) 
    return output_text

def remove_punctuations(array):
    punctuations = [",", ":", ";", "!", "?", "-", ".", "'", "&", "-"]
    array_copy = []
    for x in array:
        if x not in punctuations:
            array_copy.append(x)
    return array_copy

#Read Shakespeartext remove stopwords and punctuations
text = nltk.corpus.gutenberg.words('shakespeare-macbeth.txt')
without_stop_words = remove_stopewords(text)
without_stop_words_and_punctuations = remove_punctuations(without_stop_words)

#Search for Bi-Grams and print them out
listBigrams = nltk.bigrams(without_stop_words_and_punctuations)
freq_bi = nltk.FreqDist(listBigrams)
for k,v in freq_bi.items():
    print(k,v)

#Search for Tri-Grams and print them out
listTrigrams = nltk.trigrams(without_stop_words_and_punctuations)
freq_tri = nltk.FreqDist(listTrigrams)
for k,v in freq_tri.items():
    print(k,v)


#Read IMBD-Files and calculate/print the TF-IDF -Values
def get_tfidf(docs, ngram_range=(2,2), index=None):
    vect = TfidfVectorizer(stop_words="english", ngram_range=ngram_range)
    tfidf = vect.fit_transform(documents).todense()
    return pd.DataFrame(tfidf, columns=vect.get_feature_names(), index=index).T

documents = [open("./ressources/IMDB_1.txt", "r+").read() ,open("./ressources/IMDB_2.txt", "r+").read() ,open("./ressources/IMDB_3.txt", "r+").read() ]
documents_names = ['Doc {:d}'.format(i) for i in range(len(documents))]
print(get_tfidf(documents,index=documents_names))