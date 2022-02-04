import json


with open("./senticap_dataset.json") as f:
    senticap = json.load(f)

new_pos_train = {"images": [], "annotations": []}
new_pos_val = {"images": [], "annotations": []}
new_pos_test = {"images": [], "annotations": []}

new_neg_train = {"images": [], "annotations": []}
new_neg_val = {"images": [], "annotations": []}
new_neg_test = {"images": [], "annotations": []}

for image in senticap["images"]:    
    new_image = {}
    new_image["id"] = int(image["filename"].split(".")[0].split("_")[2])
    new_image["file_name"] = image["filename"]
    
    has_pos = False
    has_neg = False
    
    for c in image["sentences"]:
        new_annotation = {}
        new_annotation["image_id"] = int(image["filename"].split(".")[0].split("_")[2])
        new_annotation["caption"] = c["raw"]
        sentiment = c["sentiment"]
        
        if(sentiment):
            has_pos = True
            if(image["split"] == "train"):
                new_pos_train["annotations"].append(new_annotation)
            elif(image["split"] == "val"):
                new_pos_val["annotations"].append(new_annotation)
            else:
                new_pos_test["annotations"].append(new_annotation)
        else:
            has_neg = True
            if(image["split"] == "train"):
                new_neg_train["annotations"].append(new_annotation)
            elif(image["split"] == "val"):
                new_neg_val["annotations"].append(new_annotation)
            else:
                new_neg_test["annotations"].append(new_annotation)
            

    if(has_pos):
        if(image["split"] == "train"):
            new_pos_train["images"].append(new_image)
        elif(image["split"] == "val"):
            new_pos_val["images"].append(new_image)
        else:
            new_pos_test["images"].append(new_image)
    if(has_neg):
        if(image["split"] == "train"):
            new_neg_train["images"].append(new_image)
        elif(image["split"] == "val"):
            new_neg_val["images"].append(new_image)
        else:
            new_neg_test["images"].append(new_image)


print(len(new_pos_train["images"]), len(new_pos_train["annotations"]))
print(len(new_neg_train["images"]), len(new_neg_train["annotations"]))

print(len(new_pos_val["images"]), len(new_pos_val["annotations"]))
print(len(new_neg_val["images"]), len(new_neg_val["annotations"]))

print(len(new_pos_test["images"]), len(new_pos_test["annotations"]))
print(len(new_neg_test["images"]), len(new_neg_test["annotations"]))

with open('senticap_train_pos.json', 'w') as outfile:
    json.dump(new_pos_train, outfile)
with open('senticap_train_neg.json', 'w') as outfile:
    json.dump(new_neg_train, outfile)
    
with open('senticap_val_pos.json', 'w') as outfile:
    json.dump(new_pos_val, outfile)
with open('senticap_val_neg.json', 'w') as outfile:
    json.dump(new_neg_val, outfile)
    
with open('senticap_test_pos.json', 'w') as outfile:
    json.dump(new_pos_test, outfile)
with open('senticap_test_neg.json', 'w') as outfile:
    json.dump(new_neg_test, outfile)
  
              



