from nltk import tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.util import pr
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def remove_stopewords(word_tokens):
    stop_words = set(stopwords.words("english"))
        #print(stop_words)
    output_text = [w for w in word_tokens if not w.lower() in stop_words]
    output_text = []
    for w in word_tokens:
        if w not in stop_words:
            output_text.append(w) 
    return output_text

def tokenize_with_whitespace(input):
    tokenizerWhitespace = RegexpTokenizer('\w+')
    return tokenizerWhitespace.tokenize(input)

def concatenate(array):
    new = ""
    for x in array:
        new += x + " "
    return new    

# Tokenize + remove Stopwords + concatenate to one sentence/string.
IMDB1 = "Sheriff Deputy Rick Grimes wakes up from a coma to learn the world is in ruins, and must lead a group of survivors to stay alive."
IMDB2 = "The communities join forces to restore a bridge that will facilitate communication and trade; someone is gravely injured at the construction site."
IMDB3 = "Rick and his group make a risky run into Washington, D.C. to search for artifacts they will need to build the civilization he and Carl envisioned."

withoutStop_IMBD1 = concatenate(remove_stopewords(tokenize_with_whitespace(IMDB1)))
withoutStop_IMBD2 = concatenate(remove_stopewords(tokenize_with_whitespace(IMDB2)))
withoutStop_IMBD3 = concatenate(remove_stopewords(tokenize_with_whitespace(IMDB3)))

documents = [withoutStop_IMBD1, withoutStop_IMBD2, withoutStop_IMBD3]
vectorizer = CountVectorizer()
print(vectorizer.fit_transform(documents).todense())
print(vectorizer.vocabulary_)


# TF-IDF - TermFrequency Inverse Document Frequency
#Create Vector space
tfidf = TfidfVectorizer()
#Compute TF-IDF Values
result = tfidf.fit_transform(documents)
#Show vocabulary
print('\nWord indexes:')
print(tfidf.vocabulary_)
#Show IDF Values
print('\nidf values:')
for ele1, ele2 in zip (tfidf.get_feature_names(), tfidf.idf_):
    print(ele1, ":", ele2)
#Show TF-IDF Values
print("\ntf-idf value:")
print(result)
#Show in Matrix
print('\ntf-idf values in matrix form:')
print(result.toarray())