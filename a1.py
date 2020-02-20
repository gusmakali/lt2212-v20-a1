import os
import sys
import pandas as pd
import numpy as np
import numpy.random as npr
from glob import glob
from collections import Counter
# for different classifiers 
#from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split
from sklearn import datasets
#from sklearn.naive_bayes import GaussianNB
#from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

# DO NOT CHANGE THE SIGNATURES OF ANY DEFINED FUNCTIONS.
# YOU CAN ADD "HELPER" FUNCTIONS IF YOU LIKE.

def word_counts_n(txt, n=100):
    to_df = []
    for count in Counter(txt.lower().split(" ")).most_common():
        if count[1] > n and count[0].isalpha():
            to_df.append(count)

    return to_df
    
def part1_load(folder1, folder2, n=1):
    # CHANGE WHATEVER YOU WANT *INSIDE* THIS FUNCTION.

    files_dir1 = glob("{}/*.txt".format(folder1))
    files_dir2 = glob("{}/*.txt".format(folder2))

    allfiles = files_dir1 + files_dir2
    
    # create names for colums
    column = {}
    filenames = []
    classnames = []

    column["class"] = []
    column["filename"] = []

    wordcounts = {}

    for f in allfiles:
        with open (f, "r") as doc:
            word_n = word_counts_n(doc.read(), n) 

        if len(word_n) == 0:
            continue
        
        for word in word_n:
            if f not in wordcounts:
                wordcounts[f] = {}
            wordcounts[f][word[0]] = word[1]
            if word[0] not in column:
                column[word[0]] = []
    for f in allfiles:
        with open (f, "r") as doc:
            word_n = word_counts_n(doc.read(), n)
        for c in column:
            if c == "filename":
                filenames.append(f)
            if c == "class":
                if f in files_dir1:
                    classnames.append(folder1)
                else:
                    classnames.append(folder2)

            if f in wordcounts and c in wordcounts[f]:
                column[c].append(wordcounts[f][c])
            else:
                column[c].append(0)

    column["class"] = classnames
    column["filename"] = filenames
    df = pd.DataFrame(column)
    
    return df

def part2_vis(df, n=1):
    assert isinstance(df, pd.DataFrame)

    maxes = {}

    for c in df.columns.values[2:]:
        maxes[c] = df[c].max()
    maxes = {k: v for k, v in sorted(maxes.items(), key=lambda item: item[1], reverse=True)}
    m = list(maxes.keys())[:n]
    
    return pd.pivot_table(df, columns=["class"], values=m, aggfunc=np.max).plot(kind="bar")

def tfidf(tf, totaldocs, docswithword):
    return tf * (np.log10(totaldocs / docswithword) + 1)

def doc_count(df, word):
    count = 0
    for i in df[word]:
        if i != 0:
            count += 1
    return count

def part3_tfidf(df): 
    assert isinstance(df, pd.DataFrame)
    total = len(df.index)
    
    return df.transform(lambda c: tfidf(c, total, doc_count(df, c.name)) if c.name not in ["class", "filename"] else c)

def distribute_train(data):
    target = data['class']
    cols = [col for col in data.columns if col not in ['class','filename']]
    dt = data[cols]
    
    data_train, data_test, target_train, target_test = train_test_split(dt,target, test_size = 0.30, random_state = 20)

    model = SVC(C=1.5,kernel="rbf", degree=3, random_state=0)
    pred = model.fit(data_train, target_train)
    clf = pred.predict(data_test)
    return accuracy_score(target_test, clf)

    #neigh = KNeighborsClassifier(n_neighbors=3)
    #neigh.fit(data_train, target_train)
    #pred = neigh.predict(data_test)
    #return accuracy_score(target_test, pred)
    
    #gnb = GaussianNB()
    #pred = gnb.fit(data_train, target_train).predict(data_test)
    #return accuracy_score(target_test, pred, normalize = True)


df = part1_load("crude", "grain", 3)
# print(df)
# print(part3_tfidf(df))
# print(doc_count(df, "venezuela"))
print("tf   ", distribute_train(df))
print("tfidf", distribute_train(part3_tfidf(df)))