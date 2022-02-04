from __future__ import division
import operator

import numpy as np

import re
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

import cPickle
import json
import csv


def read_wordforms(path_wordforms):
    wordforms = {}
    with open(path_wordforms, "r") as wordforms_file:
        reader = csv.DictReader(
            wordforms_file, delimiter="\t", fieldnames=["class_name", "words"]
        )
        for row in reader:
            wordforms[row["class_name"]] = row["words"].split(",")        
    return wordforms


def preprocess_coco_annots(coco_annots, to_senti=False):
    PUNCTUATIONS = [
        "''", "'", "``", "`", "(", ")", "{", "}",
        ".", "?", "!", ",", ":", "-", "--", "...", ";"
    ]

    captions = coco_annots["annotations"]
    result = {}
    for e in captions:
        coco_id = e["image_id"]
        if(to_senti):
            coco_id = coco2senti[coco_id]
        
        caption = str(e["caption"].lower().strip())
        tokens = word_tokenize(caption)
        tokens = [ct for ct in tokens if ct not in PUNCTUATIONS]
        
        caption = " ".join(tokens)
        
        if coco_id in result:
            result[coco_id].append(caption)
        else:
            result[coco_id] = [caption]
            
    return result

split = "test"
sentiment = "neg"
updown_res_path = ""


coco2senti = cPickle.load(open("./coco2senti.pik", "rb"))
senti2coco = cPickle.load(open("./senti2coco.pik", "rb"))

coco_annots = json.load(open("/path/to/MSCOCO/coco/annotations/captions_val2017.json", "rb"))
senticap_annots = json.load(open("/path/to/Senticap/data/senticap_" + split + "_" + sentiment + ".json", "rb"))
gts_new = preprocess_coco_annots(senticap_annots, True)


senticap_annots_neg = json.load(open("./senticap_test_neg.json", "rb"))
gts_new_neg = preprocess_coco_annots(senticap_annots_neg, True)

senticap_annots_pos = json.load(open("./senticap_test_pos.json", "rb"))
gts_new_pos = preprocess_coco_annots(senticap_annots_pos, True)

gts_special = {}
for k, v in gts_new_pos.items():
    if k in gts_new_neg:
        if sentiment == "pos":
            gts_special[k] = gts_new_pos[k]
        elif sentiment == "neg":
            gts_special[k] = gts_new_neg[k]
        else:
            raise NotImplementedError()


gts = cPickle.load(open("../senticap/coco_mturk/gts_" + sentiment + ".pik", "rb"))

gts_coco2017 = preprocess_coco_annots(coco_annots)
senti_res = cPickle.load(open("../senticap/coco_mturk/res_" + sentiment + ".pik", "rb"))
updown_res = json.load(open(updown_res_path, "rb"))
senti_attrib_wordforms = read_wordforms("../updown-baseline/data/cbs/constraint_wordforms_attrib_senti.tsv")


def eval_style(gts, res, wordforms):
    
    senti_words = set([item for sublist in wordforms.values() for item in sublist])
    
    recall_counter_match = 0
    recall_counter = 0
    
    precision_counter_match = 0
    precision_counter = 0
    
    has_anp_counter = 0

    for image_id in gts:  
        gts_style_tokens = set()
        for c_gt in gts[image_id]:
            gts_style_tokens.update([token for token in c_gt.split() if token in senti_words]) 
        
        res_style_tokens = set()
        for preds in res:
            for c_res in preds[image_id]:
                res_style_tokens.update([token for token in c_res.split() if token in senti_words])
        
        if(len(res_style_tokens)):
            has_anp_counter += 1
        
        
        for token_gt in res_style_tokens:
            precision_counter += 1
            if token_gt in gts_style_tokens:
                precision_counter_match += 1

        for token_gt in gts_style_tokens:
            recall_counter += 1
            if token_gt in res_style_tokens:
                recall_counter_match += 1
                                    
    return precision_counter_match / precision_counter, recall_counter_match / recall_counter, has_anp_counter / len(gts)

def generate_ngrams(c, n):
    tokens = word_tokenize(c)
    #tokens = [ct for ct in tokens if ct not in PUNCTUATIONS]
    return list(ngrams(tokens, n))

def get_n_words(c):
    tokens = word_tokenize(c)
    #tokens = [ct for ct in tokens if ct not in PUNCTUATIONS]
    return len(tokens)



def n_gram_diversity(captions, n_gram_size=1):
    captions_sorted = {}
    for c in captions:
        if c["image_id"] not in captions_sorted:
            captions_sorted[c["image_id"]] = [c["caption"]]
        else:
            captions_sorted[c["image_id"]].append(c["caption"])
    
    n_words = {}
    unique_n_grams = {}
    
    score = 0
    
    for image_id, captions in captions_sorted.items():
        unique_n_grams["image_id"] = set()
        n_words["image_id"] = 0

        for c in captions:
            #print(c)
            n_grams = generate_ngrams(c, n_gram_size)
            n_words["image_id"] += get_n_words(c)
            
            for n_gram in n_grams:
                unique_n_grams["image_id"].add(n_gram)
      
        score += len(unique_n_grams["image_id"]) / n_words["image_id"]        
        
    return score / len(captions_sorted)

def count_word_appearances(res, wordforms):
    result = {}
    
    for words in wordforms.values():
        for word in words:
            result[word] = 0
    
    for im_id, c in res.items():
        c = c[0]
        tokens = c.split()
        for t in tokens:
            if t in result:
                result[t] += 1
                
    return sorted(result.items(), key=operator.itemgetter(1))   


def preprocess_updown_output(updown_res, gts, as_coco):
    res_temp = {}

    for e in updown_res:
        if e["image_id"] not in res_temp:
            res_temp[e["image_id"]] = [e["caption"]]
        else:
            res_temp[e["image_id"]].append(e["caption"])

    res = []
    n_caps_per_image = len(res_temp[res_temp.keys()[0]])
    for k in range(n_caps_per_image):
        res.append({}) 

    for key in gts.keys():
        try:
            for n in range(n_caps_per_image):
                if(as_coco):
                    res[n][key] = [res_temp[key][n]]
                else:
                    res[n][key] = [res_temp[senti2coco[key]][n]]
        except:
            del gts[key]
            
    return res, gts

def equalize_dict_keys(d1, d2):
    t1 = []
    for k in range(len(d1)):
        t1.append({})
    t2 = {}
    for key in d2:
        for k in range(len(d1)):
            t1[k][key] = d1[k][key]
        t2[key] = d2[key]   
    return t1, t2


# uncomment for evaluation on MSCOCO 2017 test split
#gts = gts_coco2017
#res, gts = preprocess_updown_output(updown_res, gts, as_coco=True)

res, gts = preprocess_updown_output(updown_res, gts, as_coco=False)

# uncomment for evaluation of Senticap predictions
# =============================================================================
# for key in senti_res.keys():
#     if key not in gts:
#         del senti_res[key]
# res = [senti_res]
# =============================================================================

res, gts = equalize_dict_keys(res, gts)
print(len(res))


total_ref_sentences = 0
for i in gts.keys():
    total_ref_sentences += len(gts[i])
print "Total ref sentences:", total_ref_sentences


# choose evaluation metrics
do_bleu = True
do_rouge = True
do_cider = True
do_meteor = True


bleu1_all = []
bleu2_all = []
bleu3_all = [] 
bleu4_all  = []
rouge_all = []
cider_all = []
meteor_all = []

bleu1_means = []
bleu2_means = []
bleu3_means = [] 
bleu4_means  = []
rouge_means = []
cider_means = []
meteor_means = []

for k in range(len(res)):
    if(do_bleu):
        bleu = Bleu()
        print("----------------------------------------")
        blue_mean, blue_scores = bleu.compute_score(gts, res[k])
    
        bleu1_all.append(blue_scores[0])
        bleu2_all.append(blue_scores[1])
        bleu3_all.append(blue_scores[2])
        bleu4_all.append(blue_scores[3])
        
        bleu1_means.append(blue_mean[0])
        bleu2_means.append(blue_mean[1])
        bleu3_means.append(blue_mean[2])
        bleu4_means.append(blue_mean[3])
        
        print "Bleu:"
        print "Positive:", blue_mean

    if(do_rouge):
        rouge = Rouge()
        rouge_mean, rouge_scores = rouge.compute_score(gts, res[k])
        rouge_all.append(rouge_scores)
        rouge_means.append(rouge_mean)

    
        print "Rouge:"
        print "Positive:", rouge_mean 
    
    if(do_cider):
        cider = Cider()
        cider_mean, cider_scores = cider.compute_score(gts, res[k])
        cider_all.append(cider_scores)
        cider_means.append(cider_mean)

        print "Cider:"
        print "Positive:", cider_mean
    
    if(do_meteor):
        meteor = Meteor()
        meteor_mean, meteor_scores = meteor.compute_score(gts, res[k])
        meteor_all.append(meteor_scores)
        meteor_means.append(meteor_mean)
        print "Meteor:"
        print "Positive:", meteor_mean


if(do_bleu):
    bleu1_all = np.stack(bleu1_all, axis=1)
    bleu2_all = np.stack(bleu2_all, axis=1)
    bleu3_all = np.stack(bleu3_all, axis=1) 
    bleu4_all  = np.stack(bleu4_all, axis=1)
    
    bleu1_means = np.stack(bleu1_means)
    bleu2_means = np.stack(bleu2_means)
    bleu3_means = np.stack(bleu3_means)
    bleu4_means = np.stack(bleu4_means)


if(do_rouge):
    rouge_all = np.stack(rouge_all, axis=1)
    rouge_means = np.stack(rouge_means)


if(do_cider):
    cider_all = np.stack(cider_all, axis=1)
    cider_means = np.stack(cider_means)


if(do_meteor):
    meteor_all = np.stack(meteor_all, axis=1)
    meteor_means = np.stack(meteor_means)


if(do_bleu):
    bleu1_max_idxs = np.argmax(bleu1_all, axis=1)
    bleu2_max_idxs = np.argmax(bleu2_all, axis=1)
    bleu3_max_idxs = np.argmax(bleu3_all, axis=1)
    bleu4_max_idxs = np.argmax(bleu4_all, axis=1)

if(do_rouge):
    rouge_max_idxs = np.argmax(rouge_all, axis=1)

if(do_cider):
    cider_max_idxs = np.argmax(cider_all, axis=1)
    cider_sorted = np.argsort(cider_all)
    
print(cider_all.shape)
print(cider_sorted.shape)

image_ids = list(res[0].keys())

updown_res_filtered = []

for n, row in enumerate(cider_sorted):
    image_id = image_ids[n]
    updown_res_filtered.append({"image_id": image_id, "caption": res[row[-1]][image_id][0]})
    updown_res_filtered.append({"image_id": image_id, "caption": res[row[-2]][image_id][0]})
    updown_res_filtered.append({"image_id": image_id, "caption": res[row[-3]][image_id][0]})
    updown_res_filtered.append({"image_id": image_id, "caption": res[row[-4]][image_id][0]})
    updown_res_filtered.append({"image_id": image_id, "caption": res[row[-5]][image_id][0]})

if(do_meteor):
    meteor_max_idxs = np.argmax(meteor_all, axis=1)

res_bleu1, gts_bleu1 = {}, {}
res_bleu2, gts_bleu2 = {}, {}
res_bleu3, gts_bleu3 = {}, {}
res_bleu4, gts_bleu4 = {}, {}
res_rouge, gts_rouge = {}, {}
res_cider, gts_cider = {}, {}
res_meteor, gts_meteor = {}, {}


for k in range(cider_max_idxs.shape[0]):
    if(do_bleu):
        bleu1_img_id = gts.keys()[k]
        res_bleu1[bleu1_img_id] = res[bleu1_max_idxs[k]][bleu1_img_id]
        gts_bleu1[bleu1_img_id] = gts[bleu1_img_id]
        
        bleu2_img_id = gts.keys()[k]
        res_bleu2[bleu2_img_id] = res[bleu2_max_idxs[k]][bleu2_img_id]
        gts_bleu2[bleu2_img_id] = gts[bleu2_img_id]
        
        bleu3_img_id = gts.keys()[k]
        res_bleu3[bleu3_img_id] = res[bleu3_max_idxs[k]][bleu3_img_id]
        gts_bleu3[bleu3_img_id] = gts[bleu3_img_id]
        
        bleu4_img_id = gts.keys()[k]
        res_bleu4[bleu4_img_id] = res[bleu4_max_idxs[k]][bleu4_img_id]
        gts_bleu4[bleu4_img_id] = gts[bleu4_img_id]
    
    if(do_rouge):
        rouge_img_id = gts.keys()[k]
        res_rouge[rouge_img_id] = res[rouge_max_idxs[k]][rouge_img_id]
        gts_rouge[rouge_img_id] = gts[rouge_img_id]
    
    if(do_cider):
        cider_img_id = gts.keys()[k]
        res_cider[cider_img_id] = res[cider_max_idxs[k]][cider_img_id]
        gts_cider[cider_img_id] = gts[cider_img_id]
   
    if(do_meteor):
        meteor_img_id = gts.keys()[k]
        res_meteor[meteor_img_id] = res[meteor_max_idxs[k]][meteor_img_id]
        gts_meteor[meteor_img_id] = gts[meteor_img_id]


if(do_bleu):
    bleu1_mean  = bleu.compute_score(gts_bleu1, res_bleu1)[0][0]
    bleu2_mean = bleu.compute_score(gts_bleu2, res_bleu2)[0][1]
    bleu3_mean = bleu.compute_score(gts_bleu3, res_bleu3)[0][2]
    bleu4_mean = bleu.compute_score(gts_bleu4, res_bleu4)[0][3]

if(do_rouge):
    rouge_mean = rouge.compute_score(gts_rouge, res_rouge)[0]

if(do_cider):
    cider_mean = cider.compute_score(gts_cider, res_cider)[0]

if(do_meteor):
    meteor_mean = meteor.compute_score(gts_meteor, res_meteor)[0]


print("input:", updown_res_path)
print("Div-1:", n_gram_diversity(updown_res, 1))
print("Div-2:", n_gram_diversity(updown_res, 2))

if(do_bleu):
    print "B1:", np.round(bleu1_mean * 100.0, 2)
    print "B2:", np.round(bleu2_mean * 100.0, 2)
    print "B3:", np.round(bleu3_mean * 100.0, 2)
    print "B4:", np.round(bleu4_mean * 100.0, 2)
    
    print "mean B1:", np.round(bleu1_means.mean() * 100.0, 2)
    print "mean B2:", np.round(bleu2_means.mean() * 100.0, 2)
    print "mean B3:", np.round(bleu3_means.mean() * 100.0, 2)
    print "mean B4:", np.round(bleu4_means.mean() * 100.0, 2)
    

if(do_rouge):
    print "rouge:", np.round(rouge_mean * 100.0, 2)
    print "mean rouge:", np.round(rouge_means.mean() * 100.0, 2)


if(do_cider):
    print "cider:", np.round(cider_mean * 100.0, 2)
    print "mean cider:", np.round(cider_means.mean() * 100.0, 2)


if(do_meteor):
    print "meteor:", np.round(meteor_mean * 100.0, 2)
    print "mean meteor:", np.round(meteor_means.mean() * 100.0, 2)


print("top5 Div-1:", n_gram_diversity(updown_res_filtered, 1))
print("top5 Div-2:", n_gram_diversity(updown_res_filtered, 2))

senti_prec, senti_rec, has_anp = eval_style(gts, res, senti_attrib_wordforms)

print("senti_prec:", senti_prec, "senti_rec:", senti_rec, "has_anp:", has_anp)
