import nltk, re, string, collections
from nltk.util import ngrams
from nltk.corpus import stopwords, shakespeare
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import gutenberg

def remove_stopewords(word_tokens):
    stop_words = set(stopwords.words("english"))
        #print(stop_words)
    output_text = [w for w in word_tokens if not w.lower() in stop_words]
    output_text = []
    for w in word_tokens:
        if w not in stop_words:
            output_text.append(w) 
    return output_text




text = nltk.corpus.gutenberg.words('shakespeare-macbeth.txt')
#text_as_String = " ".join(str(x) for x in text)
#print(text_as_String)
without_stop_words = remove_stopewords(text)
punctuations = [",", ":", ";", "!", "?"]
for x in without_stop_words:
    if x in punctuations:
        

for i in range(100):
    print(without_stop_words[i])


