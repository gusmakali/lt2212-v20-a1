# LT2212 V20 Assignment 1

Currently the program prints pandas dataframes and accuracy scores for words appearing more than 3 times in the text files (n=3).
The first part uses a helper function which counts words and words appearing more than n times.

Two helper functions are also used to calculate tf-idf according to the formula and count term frequency ("raw count").

A bonus part "distribute_train" currently uses SVM classifier as I think it works the best with this data. However, the commented part contains code for two more classifiers - GausianNB and KNeighbours. To run them just uncomment parts in distribute_train as well as imports.

All of the classifiers show pretty good score for accuracy. The SVM classifier gives 80% accuracy for part1 dataframe, and 84% for tf-idf.
Logically, tf-idf shows a bit better score. However, depending on the n value in part1 (n is how many times the words need to be appearing in the text at minimum), the ratio may change and in some cases classifiers show almost the same accurasy for both tf and tfidf. 
