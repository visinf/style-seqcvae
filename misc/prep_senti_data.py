import random
import copy
import os
import json
import csv

import nltk
from nltk.tokenize import word_tokenize

random.seed(0)

PUNCTUATIONS = [
    "''", "'", "``", "`", "(", ")", "{", "}",
    ".", "?", "!", ",", ":", "-", "--", "...", ";"
]

wordforms_objects_tsvpath = "./wordforms_objects_senti.tsv"
wordforms_objects_tsvpath_pos = "./wordforms_objects_senti_pos.tsv"
wordforms_objects_tsvpath_neg = "./wordforms_objects_senti_neg.tsv"
wordforms_attribs_tsvpath = "./wordforms_attribs_senti.tsv"

senti_pos_train_jsonpath = "./senticap/senticap_train_pos.json"
senti_pos_val_jsonpath = "./senticap/senticap_val_pos.json"
senti_pos_test_jsonpath = "./senticap/senticap_test_pos.json"

senti_neg_train_jsonpath = "./senticap/senticap_train_neg.json"
senti_neg_val_jsonpath = "./senticap/senticap_val_neg.json"
senti_neg_test_jsonpath = "./senticap/senticap_test_neg.json"

coco_train_jsonpath = "/path/to/captions.json"


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

wordforms_objects_pos = {}
with open(wordforms_objects_tsvpath_pos, "r") as wordforms_file:
    reader = csv.DictReader(wordforms_file, delimiter="\t", fieldnames=["class_name", "words"])
    for row in reader:
        word_class = {"counts": 0, "words": {}}
        for word in row["words"].split(","):
            word_class["words"][word] = 0
        wordforms_objects_pos[row["class_name"]] = word_class
        
wordforms_objects_neg = {}
with open(wordforms_objects_tsvpath_neg, "r") as wordforms_file:
    reader = csv.DictReader(wordforms_file, delimiter="\t", fieldnames=["class_name", "words"])
    for row in reader:
        word_class = {"counts": 0, "words": {}}
        for word in row["words"].split(","):
            word_class["words"][word] = 0
        wordforms_objects_neg[row["class_name"]] = word_class

with open(senti_pos_train_jsonpath) as f:
    senti_pos_train = json.load(f)['annotations']
with open(senti_pos_val_jsonpath) as f:
    senti_pos_val = json.load(f)['annotations']
with open(senti_pos_test_jsonpath) as f:
    senti_pos_test = json.load(f)['annotations']
with open(senti_neg_train_jsonpath) as f:
    senti_neg_train = json.load(f)['annotations']
with open(senti_neg_val_jsonpath) as f:
    senti_neg_val = json.load(f)['annotations']
with open(senti_neg_test_jsonpath) as f:
    senti_neg_test = json.load(f)['annotations']
    
with open(coco_train_jsonpath) as f:
    coco_train = json.load(f)['annotations']


with open("captions_new2017_balanced.json") as f:
    coco_train = json.load(f)

pos = 0
neg = 0

for c in coco_train:
    if(c["sentiment"] == 1):
        pos += 1
    else:
        print(c)
        neg += 1


def analyze_senticap(captions, wordforms_objects, wordforms_attribs):
    counter1 = 0
    counter2 = 0
    
    counter = 0
    
    attribs_per_object = {"pos": {}, "neg": {}}
    
    for c in captions:
        counter += 1
        
        if(counter % 10000 == 0):
            print(counter, "/", len(captions))
        
        c = c['caption'].lower().strip()
        caption_tokens = word_tokenize(c)
        caption_tokens = [ct for ct in caption_tokens if ct not in PUNCTUATIONS]
        
        for wordform_obj in wordforms_objects.keys():
            skip_next = False
            for word_obj in wordforms_objects[wordform_obj]['words']:
                if skip_next:
                    skip_next = False
                    continue
                
                if(word_obj in caption_tokens):
                    wordforms_objects[wordform_obj]['counts'] += 1
                    wordforms_objects[wordform_obj]['words'][word_obj] += 1
                    
                    idx = caption_tokens.index(word_obj)
                    
                    attrib1 = None
                    attrib2 = None
                    
                    if(idx > 0):
                        attrib1 = caption_tokens[idx-1]
                    if(idx > 1):
                        attrib2 = caption_tokens[idx-2]

                    try:
                        wordforms_attribs["pos"]["words"][attrib1] += 1
                        skip_next = True
                        counter1 += 1
                        
                        if wordform_obj not in attribs_per_object["pos"]:
                            attribs_per_object["pos"][wordform_obj] = {}
                        if(attrib1 in attribs_per_object["pos"][wordform_obj]):
                            attribs_per_object["pos"][wordform_obj][attrib1] += 1
                        else:
                            attribs_per_object["pos"][wordform_obj][attrib1] = 1
                            
                            
                    except:
                        try:
                            wordforms_attribs["neg"]["words"][attrib1] += 1
                            skip_next = True
                            counter1 += 1
                            
                            if wordform_obj not in attribs_per_object["neg"]:
                                attribs_per_object["neg"][wordform_obj] = {}
                            if(attrib1 in attribs_per_object["neg"][wordform_obj]):
                                attribs_per_object["neg"][wordform_obj][attrib1] += 1
                            else:
                                attribs_per_object["neg"][wordform_obj][attrib1] = 1
                            
                        except:
                            continue
                    
                    try:
                        wordforms_attribs["pos"]["words"][attrib2] += 1
                        skip_next = True
                        counter2 += 1
                        
                        if wordform_obj not in attribs_per_object["pos"]:
                            attribs_per_object["pos"][wordform_obj] = {}
                        if(attrib2 in attribs_per_object["pos"][wordform_obj]):
                            attribs_per_object["pos"][wordform_obj][attrib2] += 1
                        else:
                            attribs_per_object["pos"][wordform_obj][attrib2] = 1
                    except:
                        try:
                            wordforms_attribs["neg"]["words"][attrib2] += 1
                            skip_next = True
                            counter2 += 1
                            
                            if wordform_obj not in attribs_per_object["neg"]:
                                attribs_per_object["neg"][wordform_obj] = {}
                            if(attrib2 in attribs_per_object["neg"][wordform_obj]):
                                attribs_per_object["neg"][wordform_obj][attrib2] += 1
                            else:
                                attribs_per_object["neg"][wordform_obj][attrib2] = 1
                        except:
                            pass

            
    print("counter1:", counter1, "/", len(captions))
    print("counter2:", counter2, "/", len(captions))

    wordforms_attribs["pos"]["words"] = sorted(wordforms_attribs["pos"]["words"].items(), key=lambda item: item[1], reverse=True)
    wordforms_attribs["neg"]["words"] = sorted(wordforms_attribs["neg"]["words"].items(), key=lambda item: item[1], reverse=True)
                    
    wordforms_objects = sorted(wordforms_objects.items(), key=lambda item: item[1]["counts"], reverse=True)
    
    
    return wordforms_objects, wordforms_attribs, attribs_per_object


def generate_balanced_dataset(captions, attribs_per_obj, wordforms_objects, attrib_blacklist):
    print("WARNING: TO GENERATE MODIFIED CAPTIONS ENSURE THAT THE CORRECT WORDFORM FILES ARE USED! _constraint_....")
    captions_new = []
    captions_factual = []

    skip_tags = ["NN", "JJ", "RB"]
    break_tags = ["ATTRIB"]

    counter = 0

    for caption in captions:
        counter += 1
        
        if(counter % 10000 == 0):
            print(counter, "/", len(captions))
        
        c = caption["caption"].lower().strip()
        caption_tokens = word_tokenize(c)
        caption_tokens = [ct for ct in caption_tokens if ct not in PUNCTUATIONS and ct not in attrib_blacklist]
        caption_pos_tags = [pos[1] for pos in nltk.pos_tag(caption_tokens)]

        caption_modified = False

        for obj in attribs_per_obj.keys():
                for w in wordforms_objects[obj]["words"]:                                      
                    if(w in caption_tokens):
                        idx = caption_tokens.index(w)

                        attribs = [k for k in attribs_per_obj[obj].keys()]
                        probs = []
                        for a in attribs:
                            if (a not in caption_tokens):
                                probs.append(1)
                            else:
                                probs.append(0)
                        
                        sample = random.choices(attribs, probs)[0]

                        sample_cleaned = sample.split(" ")[-1]
                        if not sample_cleaned:
                            sample_cleaned = sample.split(" ")[-2]
                        
                        add_att = sample
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
                                
                                added_at = idx
                                
                            caption_modified = True
                            
                            if(added_at>0 and add_att[0] in ["a", "e", "i", "o", "u"] and caption_tokens[added_at-1]=="a"):
                                caption_tokens[added_at-1]="an"
                        
        if(caption_modified):
            captions_new.append({"image_id": caption["image_id"], "caption": " ".join(caption_tokens)})
        else:
            captions_factual.append({"image_id": caption["image_id"], "caption": " ".join(caption_tokens)})
            
            
    return captions_new, captions_factual
    
        
senti_neg_train_obj, senti_neg_train_attrib, senti_neg_train_attribs_per_obj = analyze_senticap(senti_neg_train + senti_neg_val + senti_neg_test, copy.deepcopy(wordforms_objects_neg), copy.deepcopy(wordforms_attribs))
senti_pos_train_obj, senti_pos_train_attrib, senti_pos_train_attribs_per_obj = analyze_senticap(senti_pos_train + senti_pos_val + senti_pos_test, copy.deepcopy(wordforms_objects_pos), copy.deepcopy(wordforms_attribs))

captions_new_neg, captions_factual1 = generate_balanced_dataset(coco_train, 
                                         senti_neg_train_attribs_per_obj["neg"], 
                                         senti_neg_train_obj, 
                                         set(senti_pos_train_attrib["pos"]["words"].keys()))
captions_new_pos, captions_factual2 = generate_balanced_dataset(coco_train, 
                                         senti_pos_train_attribs_per_obj["pos"], 
                                         senti_pos_train_obj, 
                                         set(senti_neg_train_attrib["neg"]["words"].keys()))

for c in captions_new_neg[:200]:
    print(c)

for c in captions_new_neg:
    c["sentiment"] = -1
    
for c in captions_new_pos:
    c["sentiment"] = 1

captions_new_pos_obj, captions_new_pos_attrib, captions_new_pos_attribs_per_obj = analyze_senticap(captions_new_pos, copy.deepcopy(wordforms_objects), copy.deepcopy(wordforms_attribs))
captions_new_neg_obj, captions_new_neg_attrib, captions_new_neg_attribs_per_obj = analyze_senticap(captions_new_neg, copy.deepcopy(wordforms_objects), copy.deepcopy(wordforms_attribs))

with open(os.path.join("./", "captions_new_neg_balanced.json"), "w") as write_file:
    json.dump(captions_new_neg, write_file)

with open(os.path.join("./", "captions_new_pos_balanced.json"), "w") as write_file:
    json.dump(captions_new_pos, write_file)
    
with open(os.path.join("./", "captions_new_balanced.json"), "w") as write_file:
    json.dump(captions_new_pos + captions_new_neg, write_file)
