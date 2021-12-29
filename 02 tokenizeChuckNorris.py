from nltk import tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer



#1 Tokenize
def tokenize_with_whitespace(input):
    tokenizerWhitespace = RegexpTokenizer('\w+')
    return tokenizerWhitespace.tokenize(input)

def tokenize_with_capitwords(input):
    tokenizerCapitalWord = RegexpTokenizer('[A-Z]\w+')
    return tokenizerCapitalWord.tokenize(input)


#2 Stop Word Removal
def remove_stopewords(word_tokens):
    stop_words = set(stopwords.words("english"))
        #print(stop_words)
    output_text = [w for w in word_tokens if not w.lower() in stop_words]
    output_text = []
    for w in word_tokens:
        if w not in stop_words:
            output_text.append(w) 
    return output_text




#3 Stemming
def stemm(word_tokens):
    ps = PorterStemmer()
    word_tokens_copy = []
    for i in range(0,len(word_tokens)):
        #print(output_text[i] , " ", ps.stem(output_text[i]))
        word_tokens_copy.append(ps.stem(word_tokens[i]))
    return word_tokens_copy


#4 Lemmatization
def lemmatize(word_tokens):  
    lemmatize = WordNetLemmatizer()
    word_tokens_copy = []
    for i in range(0,len(word_tokens)):
        #print(word_tokens[i] , " ", lemmatize.lemmatize(word_tokens[i]))
        word_tokens_copy.append(lemmatize.lemmatize(word_tokens[i]))
    return(word_tokens_copy)




############# Main  #############
input = "When Alexander Graham Bell invented the telephone, he had three missed calls from Chuck Norris."
tokenized = tokenize_with_whitespace(input)
without_stopwords = remove_stopewords(tokenized)
stemmed = stemm(without_stopwords)
lemmatized = lemmatize(stemmed)

print("#0 Input:             ", input)
print("#1 Tokenized:         ", tokenized)
print("#2 Without stopwords: ", without_stopwords)
print("#3 Stemmed:           ", stemmed)
print("#4 Lemmatized:        ", lemmatized)




