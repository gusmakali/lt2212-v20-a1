import os
import sys
import pandas as pd
import numpy as np
import numpy.random as npr
from glob import glob
from collections import Counter
import math 

from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

# DO NOT CHANGE THE SIGNATURES OF ANY DEFINED FUNCTIONS.
# YOU CAN ADD "HELPER" FUNCTIONS IF YOU LIKE.

N = 20

def word_counts_n(txt, n=100):
    to_df = []
    for count in Counter(txt.lower().split(" ")).most_common():
        if count[1] > n and count[0].isalpha():
            to_df.append(count)

    return to_df
    
def part1_load(folder1, folder2):
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
            word_n = word_counts_n(doc.read(), N) 

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
            word_n = word_counts_n(doc.read(), N) 
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

# print(part1_load("crude", "grain"))

def part2_vis(df):
    assert isinstance(df, pd.DataFrame)

    maxes = {}

    for c in df.columns.values[2:]:
        maxes[c] = df[c].max()
    maxes = {k: v for k, v in sorted(maxes.items(), key=lambda item: item[1], reverse=True)}
    m = list(maxes.keys())[:5]
    
    return pd.pivot_table(df, columns=["class"], values=m, aggfunc=np.max).plot(kind="bar")

def tfidf(v, totaldocs, docswithword):
    return v * math.log(totaldocs / docswithword)

def doc_count(df, word):
    assert isinstance(df, pd.DataFrame)
    return len(df[df[word] > 0].index)

def part3_tfidf(df):

    assert isinstance(df, pd.DataFrame)
    total = len(df.index)
    return df.transform(lambda c: tfidf(c, total, doc_count(df, c.name)) if c.name not in ["class", "filename"] else c)
    
# df = part1_load("crude", "grain")
# print(part3_tfidf(df))

def part_bonus(df):
    