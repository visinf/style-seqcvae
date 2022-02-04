import os
import pickle

import numpy as np

from datasets.config_attrib_selection import attrib_selection


def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class CocoAttributesReader(object):
    
    def __init__(self, attribs_dir_path: str):
        self.attrib_weight_threshold = 0.1
        self.attrib_min_appearance = 0
        self.attribs_n_max_per_image = 200

        # ATTENTION: image2obj_insts, obj_inst2attrib_inst, attrib_inst2attrib_vector still contain elements not appearing in image list
        result_read_attributes = self.read_attributes(attribs_dir_path)
        
        self.image_ids = set(result_read_attributes[0])
        self.image2obj_insts = result_read_attributes[1]
        self.obj_inst2attrib_inst = result_read_attributes[2]
        self.attrib_inst2attrib_vector = result_read_attributes[3]
        self.ignore_attrib_indices = result_read_attributes[4]
        self.attrib_names = result_read_attributes[5]
        self.attrib_image_count = result_read_attributes[6]
        self.attrib2attrib_inst_count = result_read_attributes[7]
        
        self.n_attribs = len(self.attrib_names)

        self.att_counts = np.zeros(self.n_attribs)
        
        for k,v in self.attrib2attrib_inst_count.items():
            self.att_counts[k] = v
        
        self.obj_inst2obj_id = load_obj(os.path.join(attribs_dir_path, "obj_inst2obj_id.pkl"))
        self.obj_id2obj_name = load_obj(os.path.join(attribs_dir_path, "obj_id2obj_name.pkl"))

    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, image_id: int):
        obj_insts = self.image2obj_insts[image_id]
        
        #print(obj_insts)
        result = []
        for obj_inst in obj_insts:     
            if(obj_inst in self.obj_inst2attrib_inst):
                attrib_inst = self.obj_inst2attrib_inst[obj_inst]
                
                try:
                    attrib_vec = self.attrib_inst2attrib_vector[attrib_inst]
                    #result.append([obj_inst, attrib_vec])               # attribs as sparse arrays
                    result.append([obj_inst, list(np.nonzero(attrib_vec)[0])])   # attribs as indizes
                    #if(attrib_vec.sum() > 0):
                        #result.append([obj_inst, [self.attrib_names[x] for x in np.nonzero(attrib_vec)[0]]])   # attribs as strings
                    #result.append([obj_inst, [[self.attrib_names[x], attrib_vec[x]] for x in np.nonzero(attrib_vec)[0]]])
                except:
                    pass

        return result

    
    def filter_duplicates(self, result):
        result_filtered = {}
        
        for obj in result:
            if(obj[0] not in result_filtered):
                result_filtered[obj[0]] = obj[1]
            else:
                result_filtered_atts = [a[0] for a in result_filtered[obj[0]]]
                
                for attrib in obj[1]:
                    try:
                        idx = result_filtered_atts.index(attrib[0])
                        result_filtered[obj[0]][idx][1] = max(result_filtered[obj[0]][idx][1], attrib[1])
                    except ValueError:
                        result_filtered[obj[0]].append(attrib)
        
        return [[key, value] for key, value in result_filtered.items()]
    
    
    def read_attributes(self, attribs_dir_path, ignore_attrib_indices=None):
        attrib_inst2attrib_vector = load_obj(os.path.join(attribs_dir_path, "attrib_inst2attrib_vector.pkl"))
        attrib_inst2obj_inst = load_obj(os.path.join(attribs_dir_path, "attrib_inst2obj_inst.pkl"))
        obj_inst2attrib_inst = load_obj(os.path.join(attribs_dir_path, "obj_inst2attrib_inst.pkl"))
        
        obj_inst2image = load_obj(os.path.join(attribs_dir_path, "obj_inst2image.pkl"))
        image2obj_insts = load_obj(os.path.join(attribs_dir_path, "image2obj_insts.pkl"))

        attrib2string = load_obj(os.path.join(attribs_dir_path, "attrib2string.pkl")) 
        
        attrib_names = []
        for key in sorted(attrib2string.keys()):
            attrib_names.append(attrib2string[key])
        
# =============================================================================
#         for a in attrib_names:
#             b = a.split(" ")[-1]
#             if not b:
#                 b = a.split(" ")[-2]
#             print(b + " " + b + "  ")#,  a.split(" "))
# =============================================================================
        
        # remove ignored attributes from attribute name list
        attrib_selection_list = np.array(list(attrib_selection.values()), dtype=int)
        
        attrib_ignore_selection_idxs = np.argwhere(attrib_selection_list == 0)
        
        attrib_names = np.delete(attrib_names, attrib_ignore_selection_idxs).tolist()
        
        attrib2attrib_inst_count = {}
        attrib_image_count = {}
        attrib2images = {}
        for att_id, atts in list(attrib_inst2attrib_vector.items()):
            instance_id = attrib_inst2obj_inst[att_id]
            
            try:
                coco_id = obj_inst2image[instance_id]
            except:
                del attrib_inst2attrib_vector[att_id]
                continue
                
            
            # remove ignored attributes from attribute arrays
            atts = np.delete(atts, attrib_ignore_selection_idxs)
            #atts = (atts * attrib_selection_list)
            idxs_larger = np.argwhere(atts >= self.attrib_weight_threshold)
            idxs_larger = [idx[0] for idx in idxs_larger]
            
            idxs_too_small = atts < self.attrib_weight_threshold
            
            # set attribute values in attribute array to zero if smaller than threshold
            atts[idxs_too_small] = 0.0
            attrib_inst2attrib_vector[att_id] = atts
            
            # add larger attributes to count dict and attrib2images dict
            for idx in idxs_larger:
                if(idx not in attrib2attrib_inst_count):
                    attrib2attrib_inst_count[idx] = 1
                else:
                    attrib2attrib_inst_count[idx] += 1
                    
                if(idx not in attrib2images):
                    attrib2images[idx] = {coco_id}
                else:
                    attrib2images[idx].add(coco_id)
        
        # generate image count dict for attribute appearance
        for att_id, image_ids in attrib2images.items():
            attrib_image_count[att_id] = len(image_ids)
        
        # detect attributes with count lower than threshold
        if(ignore_attrib_indices is None):
            ignore_attrib_indices = []
            for att_id, count in attrib_image_count.items(): 
                if(count < self.attrib_min_appearance):
                    ignore_attrib_indices.append([att_id])
        elif(not ignore_attrib_indices):
            raise ValueError("no ignore_attrib_indices is given.")
        
        
        attrib_names = np.delete(attrib_names, ignore_attrib_indices).tolist()
        
        for image_id, obj_insts in image2obj_insts.items():        
            attrib_insts = []
            for obj_inst in obj_insts:
                if(obj_inst in obj_inst2attrib_inst):
                    attrib_insts.append(obj_inst2attrib_inst[obj_inst])
            
            attrib_vectors = []
            rem_list = []
            for attrib_inst in attrib_insts:
                if(attrib_inst in attrib_inst2attrib_vector):
                    attrib_vectors.append(attrib_inst2attrib_vector[attrib_inst])
                else:
                    rem_list.append(attrib_inst)
            
            for attrib_inst in rem_list:
                attrib_insts.remove(attrib_inst)
                    
        
            atts = np.sum(attrib_vectors, axis=0)            

            idxs_larger = np.argwhere(atts > 0)
            idxs_larger = [idx[0] for idx in idxs_larger]
            n_attribs = min(len(idxs_larger), self.attribs_n_max_per_image)
            
            atts_count = np.ones(atts.shape) * 99999            
            for idx in idxs_larger:
                atts_count[idx] = attrib_image_count[idx]
            
            final_attribs_idxs = np.argsort(atts_count)[:n_attribs]
            
            for attrib_inst in attrib_insts:
                atts_new  = np.zeros(atts.shape)
            
                for idx in final_attribs_idxs:
                    atts_new[idx] = attrib_inst2attrib_vector[attrib_inst][idx]
                attrib_inst2attrib_vector[attrib_inst] = atts_new
        
        # remove attributes from dicts which appear in less than config.attrib_min_appearance
        attrib2attrib_inst_count = {}
        attrib2images = {}
        for att_id, atts in attrib_inst2attrib_vector.items():
            instance_id = attrib_inst2obj_inst[att_id]
            coco_id = obj_inst2image[instance_id]

            atts = np.delete(atts, ignore_attrib_indices)            
            attrib_inst2attrib_vector[att_id] = atts

            idxs_larger = np.argwhere(atts > 0)
            idxs_larger = [idx[0] for idx in idxs_larger]
                
            for idx in idxs_larger:
                if(idx not in attrib2attrib_inst_count):
                    attrib2attrib_inst_count[idx] = 1
                else:
                    attrib2attrib_inst_count[idx] += 1
                    
                if(idx not in attrib2images):
                    attrib2images[idx] = {coco_id}
                else:
                    attrib2images[idx].add(coco_id)           

        attrib_image_count = {}
        for att_id, image_ids in attrib2images.items():
            attrib_image_count[att_id] = len(image_ids)
                
        
        # extract image id list only containing images with not ignored attributes and containing at leat one attribute
        image_ids = set(image_id for set_ in attrib2images.values() for image_id in set_)
          
        # ATTENTION: image2obj_insts, obj_inst2attrib_inst, attrib_inst2attrib_vector still contain elements not appearing in image list
        return list(image_ids), image2obj_insts, obj_inst2attrib_inst, attrib_inst2attrib_vector, ignore_attrib_indices, attrib_names, attrib_image_count, attrib2attrib_inst_count
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
