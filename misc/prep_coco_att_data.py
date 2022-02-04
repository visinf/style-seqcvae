import json
import csv

import collections

import matplotlib.pyplot as plt
from PIL import Image

from nltk.tokenize import word_tokenize
import nltk
import random
import copy

import os

random.seed(0)

PUNCTUATIONS = [
    "''", "'", "``", "`", "(", ")", "{", "}",
    ".", "?", "!", ",", ":", "-", "--", "...", ";"
]

wordforms_objects_tsvpath = "/path/to/constraint_wordforms_exp.tsv"
wordforms_attribs_tsvpath = "/path/to/constraint_wordforms_attribs_exp.tsv"
coco_train_jsonpath = "/path/to/captions.json"
attribs_jsonpath = "/path/to/attrib_detections.json"

with open(attribs_jsonpath) as f:
    image_id2attribs = json.load(f)

wordforms_attribs = {}
with open(wordforms_attribs_tsvpath, "r") as wordforms_file:
    reader = csv.DictReader(wordforms_file, delimiter="\t", fieldnames=["class_name", "words"])
    for row in reader:
        word_class = {"counts": 0, "words": {}}
        for word in row["words"].split(","):
            word_class["words"][word] = 0
        wordforms_attribs[row["class_name"]] = word_class

wordforms_objects = {}
with open(wordforms_objects_tsvpath, "r") as wordforms_file:
    reader = csv.DictReader(wordforms_file, delimiter="\t", fieldnames=["class_name", "words"])
    for row in reader:
        word_class = {"counts": 0, "words": {}}
        for word in row["words"].split(","):
            word_class["words"][word] = 0
        wordforms_objects[row["class_name"]] = word_class

with open(coco_train_jsonpath) as f:
    coco_train = json.load(f)['annotations']

def analyze_attribs():
    result = {}
    for item in image_id2attribs:
        atts = [a[0] for o in item["candidates"] for a in o[1]]
        for att in atts:
            try:
                result[att] += 1
            except:
                result[att] = 1
    return result

def analyze_dataset(captions, wordforms_objects, wordforms_attribs):
    
    obj_word2obj_wordform = {}
    for k, v in wordforms_objects.items():
        for w in v["words"]:
            if w not in obj_word2obj_wordform:
                obj_word2obj_wordform[w] = [k]
            else:
                obj_word2obj_wordform[w].append(k)
                
    attrib_word2attrib_wordform = {}
    for k, v in wordforms_attribs.items():
        for w in v["words"]:
            if w not in attrib_word2attrib_wordform:
                attrib_word2attrib_wordform[w] = [k]
            else:
                attrib_word2attrib_wordform[w].append(k)
    
    counter = 0
    
    for c in captions:
        counter += 1
        
        if(counter % 10000 == 0):
            print(counter, "/", len(captions))
        
        c = c['caption'].lower().strip()
        caption_tokens = word_tokenize(c)
        caption_tokens = [ct for ct in caption_tokens if ct not in PUNCTUATIONS]
        
        skip_next = False
        for c_word in caption_tokens:
            if skip_next:
                skip_next = False
                continue

            if c_word in obj_word2obj_wordform:
                for obj_wordform in obj_word2obj_wordform[c_word]:
                    wordforms_objects[obj_wordform]["counts"] += 1
                    wordforms_objects[obj_wordform]["words"][c_word] += 1
                    
            if c_word in attrib_word2attrib_wordform:
                for attrib_wordform in attrib_word2attrib_wordform[c_word]:
                    wordforms_attribs[attrib_wordform]["counts"] += 1
                    wordforms_attribs[attrib_wordform]["words"][c_word] += 1
                    
    
    return wordforms_objects, wordforms_attribs


def generate_balanced_dataset():
    skip_tags = ["NN", "JJ", "RB"]
    break_tags = ["ATTRIB"]

    captions_new = []        
    image_id2captions = {}
    for coco_captions in [coco_train]:
        for annot in coco_captions:
            if annot["image_id"] in image_id2captions:
                image_id2captions[annot["image_id"]].append(annot["caption"])
            else:
                image_id2captions[annot["image_id"]] = [annot["caption"]]
    
    for n, image in enumerate(image_id2attribs):
        try:
            captions = image_id2captions[image["image_id"]]
        except:
            continue
        
        attributes = image["candidates"]
    
        if(n % 1000 == 0):
            print(n, "/", len(image_id2attribs))
    
        for c in captions:
            c = c.lower().strip()
            caption_tokens = word_tokenize(c)
            caption_tokens = [ct for ct in caption_tokens if ct not in PUNCTUATIONS]
            caption_pos_tags = [pos[1] for pos in nltk.pos_tag(caption_tokens)]
    
            caption_modified = False

            for o in attributes:
                for w in wordforms_objects[o[0]]["words"]:                                      
                    if(w in caption_tokens):
                        idx = caption_tokens.index(w)
                        
                        o_a = [a[0] for a in o[1]] 
                        o_a_p = [1 / pow(att_counts[a], 2) for a in o_a]
                        sample = random.choices(o_a, o_a_p)[0]
                        
                        sample_cleaned = sample.split(" ")[-1]
                        if not sample_cleaned:
                            sample_cleaned = sample.split(" ")[-2]
                        
                        add_att = random.choice(list(wordforms_attribs[sample_cleaned]["words"].keys()))
                                                
                        if add_att not in caption_tokens:
                            added_at = -1
                            if(idx > 1 and caption_pos_tags[idx-2] in skip_tags and caption_pos_tags[idx-1] in skip_tags and
                               caption_pos_tags[idx-2] not in break_tags and
                               (idx < 3 or caption_pos_tags[idx-3] not in break_tags)):
                                caption_tokens.insert(idx-2, add_att)
                                caption_pos_tags.insert(idx-2, "ATTRIB")
                                
                                added_at = idx-2
                                
                            elif(idx and caption_pos_tags[idx-1] in skip_tags and
                               caption_pos_tags[idx-1] not in break_tags and
                               (idx < 2 or caption_pos_tags[idx-2] not in break_tags)):                                
                                caption_tokens.insert(idx-1, add_att)
                                caption_pos_tags.insert(idx-1, "ATTRIB")
                                
                                added_at = idx-1

                            elif(caption_pos_tags[idx] not in break_tags and
                                (idx < 1 or caption_pos_tags[idx-1] not in break_tags)):
                                caption_tokens.insert(idx, add_att)
                                caption_pos_tags.insert(idx, "ATTRIB")
                            
                            caption_modified = True

                            if(added_at>0 and add_att[0] in ["a", "e", "i", "o", "u"] and add_att != "useful" and caption_tokens[added_at-1]=="a"):
                                caption_tokens[added_at-1]="an"

                        
            if(caption_modified):
                captions_new.append({"image_id": image["image_id"], "caption": " ".join(caption_tokens), "attributes": attributes})
    
    return captions_new

captions_new = generate_balanced_dataset()

with open(os.path.join("./", "att_captions_new_balanced.json"), "w") as write_file:
    json.dump(captions_new, write_file)

