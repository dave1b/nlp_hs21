import csv
from nltk.util import pr


def csvToArray(filepath):
    #array with paragraphs
    dataArray = []
    # open file for reading
    with open(filepath) as csvDataFile:

        # read file as csv file 
        csvReader = csv.reader(csvDataFile)

        # for every row, print the row
        for row in csvReader:
            dataArray.append(str(row))

    #print("Length of array: ", len(dataArray))
    return  ([x.lower() for x in dataArray])