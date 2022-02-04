import os
import json
import csv
import pickle

from torchtext.vocab import GloVe
from sklearn.decomposition import PCA
import numpy as np

from config_attrib_selection import attrib_selection


temp = {}
for k,v in attrib_selection.items():
    temp[k.split(" ")[0]] = v
attrib_selection = temp

glove = GloVe(name="42B", dim=300, cache="./.vector_cache")


with open(os.path.join("./", "./senticap/wordform_sentiments.json"), "r") as read_file:
    word_sentiments = json.load(read_file)

wordforms_attribs_tsvpath = "./senticap/constraint_wordforms_attribs_exp.tsv"
wordforms_attribs = {}
with open(wordforms_attribs_tsvpath, "r") as wordforms_file:
    reader = csv.DictReader(wordforms_file, delimiter="\t", fieldnames=["class_name", "words"])
    for row in reader:
        word_class = {"counts": 0, "words": {}}
        for word in row["words"].split(","):
            # Constraint words can be "multi-word" (may have more than one tokens).
            # Add all tokens to the vocabulary separately.
            word_class["words"][word] = 0
        wordforms_attribs[row["class_name"]] = word_class


wordform_list_all = []
for k,v in attrib_selection.items():
    word = k.split(" ")[0]
    wordform_list_all.append([word, word_sentiments[word][0] - word_sentiments[word][2], v])

wordform_list_selected = [w[0] for w in wordform_list_all if w[2]]        
wordform_list_selected_glove = []
for w in wordform_list_selected:
    wordform_list_selected_glove.append(glove[w])
wordform_list_selected_glove = np.stack(wordform_list_selected_glove)

wordform_list_all = sorted(wordform_list_all, key=lambda item: item[1], reverse=False)
wordform_list_all = [w[0] for w in wordform_list_all]
wordform_list_all_glove = []
for w in wordform_list_all:
    wordform_list_all_glove.append(glove[w])
wordform_list_all_glove = np.stack(wordform_list_all_glove)


wordform_top10_neg = wordform_list_all[:10]
wordform_top10_pos = wordform_list_all[-10:]
wordform_top10_pos_glove = []
wordform_top10_neg_glove = []

for w in wordform_top10_pos:
    wordform_top10_pos_glove.append(glove[w])
for w in wordform_top10_neg:
    wordform_top10_neg_glove.append(glove[w])
wordform_top10_pos_glove = np.stack(wordform_top10_pos_glove)
wordform_top10_neg_glove = np.stack(wordform_top10_neg_glove)
both = np.concatenate((wordform_top10_pos_glove, wordform_top10_neg_glove), axis=0)
    
word_list_all = []
word_list_all_glove = []
word_list_selected = []
word_list_selected_glove = []
for k, v in wordforms_attribs.items():
    try:
        if attrib_selection[k]:
            word_list_selected.extend(list(wordforms_attribs[k]["words"].keys()))
        word_list_all.extend(list(wordforms_attribs[k]["words"].keys()))
    except:
        pass
word_list_selected = [w for w in word_list_selected if w not in wordform_list_selected]

for w in word_list_all:
    word_list_all_glove.append(glove[w])
for w in word_list_selected:
    word_list_selected_glove.append(glove[w])
word_list_all_glove = np.stack(word_list_all_glove)
word_list_selected_glove = np.stack(word_list_selected_glove)

n_components = 10
pca = PCA(n_components=n_components)
pca.fit(both)
pca_wordform_list_all = pca.transform(wordform_list_all_glove)
pca_wordform_list_selected = pca.transform(wordform_list_selected_glove)
pca_word_list_selected = pca.transform(word_list_selected_glove)

wordlist = wordform_list_all
pca_list = pca_wordform_list_all

glove_n = dict(zip(wordlist, pca_list))
with open("sentiglove" + str(n_components) + ".pkl", "wb") as f:
    pickle.dump(glove_n, f)