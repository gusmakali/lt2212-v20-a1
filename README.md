# LT2212 V20 Assignment 1

Currently the program prints pandas dataframes and accuracy scores for words appearing more than 300 times in all corpus (n=300).
The first part uses a helper function which counts words per document. 

Three small (one step) helper functions are also used to calculate tf-idf according to the formula: one finction is a general furmula for tfidf and another one calculates a number of documents with word. The third one is used to extract idf per corpus.

A bonus part "distribute_train" currently uses KNeigbours classifier as I think it works the best with this data. However, the commented part contains code for two more classifiers - GausianNB and SVC. To run them just uncomment parts in distribute_train as well as imports.

All of the classifiers show pretty good score for accuracy. The KNeigh classifier gives 80-90% accuracy for both dataframes, depending on the n value in part1. The accuracy falls down when classifier is run on tfidf (part3 df) with N being a small number, and grows when N is a bigger value. Vice versa, for bigger N there is lower accuracy when run on part1 dataframe, and with smaller N accuracy is higher.

