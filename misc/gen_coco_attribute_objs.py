import json
import pickle

from torchvision.datasets import CocoCaptions


# --------- COCO ---------
path_mscoco_train_images = "/path/to/MSCOCO/coco/images/train2017/"
path_mscoco_train_captions = "/path/to/MSCOCO/coco/annotations/captions_train2017.json"
path_mscoco_train_instances = "/path/to/MSCOCO/coco/annotations/instances_train2017.json"

path_mscoco_val_images = "/path/to/MSCOCO/coco/images/val2017/"
path_mscoco_val_captions = "/path/to/MSCOCO/coco/annotations/captions_val2017.json"
path_mscoco_val_instances = "/path/to/MSCOCO/coco/annotations/instances_val2017.json"

path_mscoco_test_images = "/path/to/MSCOCO/coco/images/test2014/"


# --------- COCOAttributes ---------
path_coco_attributes = "./cocottributes_eccv_version.pkl"



def save_obj(obj, name):
    with open('./obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(name):
    with open('./obj/' + name, 'rb') as f:
        return pickle.load(f)



def extract_instance_information(coco_instances_train_json, coco_instances_val_json):
    obj_id2obj_name = {}
    
    image2obj_insts = {}
    obj_inst2image = {}
    obj_inst2obj_id = {}
    obj_id2images = {}

    instance_files = [coco_instances_train_json, coco_instances_val_json]
    
    for obj in coco_instances_train_json["categories"]:
        obj_id2obj_name[obj["id"]] = obj["name"]

    for instance_file in instance_files:
        counter = 0
        for inst in instance_file["annotations"]:
            image_id = inst["image_id"]
            obj_inst = inst["id"]
            obj_id = inst["category_id"]
    
            if obj_id in obj_id2images:
                obj_id2images[obj_id].append(image_id)
            else:
                obj_id2images[obj_id] = [image_id]
            
            if image_id in image2obj_insts:
                image2obj_insts[image_id].append(obj_inst)
            else:
                image2obj_insts[image_id] = [obj_inst]
                
            obj_inst2image[obj_inst] = image_id
            obj_inst2obj_id[obj_inst] = obj_id
            counter += 1
            
            if(counter % 10000 == 0):
                print(counter)

    save_obj(obj_id2obj_name, "obj_id2obj_name")

    save_obj(image2obj_insts, "image2obj_insts")
    save_obj(obj_inst2image, "obj_inst2image")
    save_obj(obj_inst2obj_id, "obj_inst2obj_id")
    save_obj(obj_id2images, "obj_id2images")
    
    print("Done.")
        

with open(path_mscoco_train_instances, 'r') as f:
    coco_instances_train_json = json.load(f)
    
with open(path_mscoco_val_instances, 'r') as f:
    coco_instances_val_json = json.load(f)

extract_instance_information(coco_instances_train_json, coco_instances_val_json)

coco_captions_train = CocoCaptions(root = path_mscoco_train_images, 
                                         annFile = path_mscoco_train_captions, 
                                         transform=None)

coco_captions_val = CocoCaptions(root = path_mscoco_val_images, 
                                         annFile = path_mscoco_val_captions, 
                                         transform=None)

with open(path_coco_attributes, 'rb') as f:
    coco_attribs_json = pickle.load(f, encoding='latin1')


attrib2instance = load_obj("attrib2instance.pkl")
attrib2vector = coco_attribs_json["ann_vecs"]
attrib2split = coco_attribs_json["split"]

train_count = 0
val_count = 0

train = []
val = []

for key, value in attrib2split.items():
    if(value == "train2014"):
        attrib2split[key] = 0
        train.append(key)
        train_count += 1
    elif(value == "val2014"):
        attrib2split[key] = 1
        val.append(key)
        val_count += 1

split2attrib = coco_attribs_json["split"]

instance2attrib = dict([(value, key) for key, value in attrib2instance.items()]) 

save_obj({"train": train, "val": val}, "instance2attrib")
save_obj(instance2attrib, "instance2attrib")
save_obj(attrib2vector, "attrib2vector")
save_obj(attrib2split, "attrib2split")

    


 