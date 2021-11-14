import gensim
import time
from nltk import tag
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from csvToArray import csvToArray



def csvToDoc2Vec(csvPath, pathForModelSave):
    startTime = time.time()
    dataSet = csvToArray(csvPath)
    print("Data received")

    # Tokenize Exercise Data & Apoint Paraghraph ID = Tags
    print("Tagging data")
    taggedData = [TaggedDocument(words=word_tokenize(_d), tags=[str(i)]) for i, _d in enumerate(dataSet)]

    # Initialize Doc2Vec
    print("initializing Doc2Vec")
    model = gensim.models.doc2vec.Doc2Vec(vector_size=30, min_count = 2, epochs=120)

    # Build Vocabulary
    print("Build Voc")
    model.build_vocab(taggedData)

    # Train Model
    print("train model")
    model.train(taggedData, total_examples=model.corpus_count,epochs=120)

    # Save Model
    print("save model")
    model.save(pathForModelSave)
    print("Duration of building Doc2Vec: ", time.time()-startTime)

def main():
    csvToDoc2Vec('Doc2vec/datasets/shortjokes.csv',"Doc2Vec/trained/jokes.model")
    #example()

if __name__ == '__main__':
    main()