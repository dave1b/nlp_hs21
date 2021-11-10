from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from csvToArray import csvToArray

# Load Model
model = Doc2Vec.load("Doc2vec/trained/jokes.model")
dataSet = csvToArray('Doc2vec/datasets/shortjokes.csv')

def main():
    userInputSimilar()
    #example()

def userInputSimilar():
    joke = input("Welcome to Joke-Finder!!! \n---------------------------- \n Enter your favorite joke: ")
    print("You entered: ", joke, "\nThis are the 6 most similar jokes:")
    findAndPrintSimilarJoke(joke)

def userInputSimilar():
    joke = input("Welcome to Joke-Finder!!! \n---------------------------- \n Enter your favorite joke: ")
    print("You entered: ", joke, "\nThis are the 6 most similar jokes:")
    findAndPrintSimilarJoke(joke)

def findAndPrintSimilarJoke(joke):
    joke = word_tokenize(joke.lower())
    similarJokes = model.dv.most_similar(positive=[model.infer_vector(joke)], topn=6)
    for joke in similarJokes:
        print(dataSet[int(joke[0])] , ' has the similarity: ', joke[1])

#Example
def example():
    joke = "Never trust math teachers who use graph paper. They're always plotting something."
    findAndPrintSimilarJoke(joke)

if __name__ == '__main__':
    main()