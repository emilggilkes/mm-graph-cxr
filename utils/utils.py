import os
import pandas as pd
import json
import numpy as np
from tqdm import tqdm

# rootdir = '/home/agun/mimic/dataset/VG/FeatureData/'
# outputdir = "/home/agun/mimic/dataset/VG/"
# files = os.listdir(rootdir)

# coco_data = pd.DataFrame(
#                     {'image_id': files
#                     })
# coco_data.to_csv(os.path.join(outputdir, "new_train.csv"), sep='\t', index=False)
# print(coco_data.head(5))

def get_class_weights(data_dir, image_ids, output_dir, filename):
    object_counts = {i:0 for i in range(18)}
    class_counts = np.zeros(shape=(18,9))

    for id in tqdm(image_ids):
        #print(id)
        with open(os.path.join(data_dir, id+".json"), 'r') as f:
            image_data = json.load(f)
        class_counts += np.array(image_data['target'])
        image_features = np.array(image_data['features'])
        feat_sum = image_features.sum(axis=1)
        objs_idx = np.where(feat_sum!=0)[0]
        for i in objs_idx:
            object_counts[i] +=1
    
    class_counts = pd.DataFrame(data=class_counts, columns=list(range(9)))
    
    object_counts = pd.Series(object_counts, index=list(range(18)))
    # print(object_counts)
    #class_weights = class_counts / object_counts
    class_counts.to_csv(os.path.join(output_dir, filename), index=False)
    object_counts.to_csv(os.path.join(output_dir, "object_counts.csv"), index=False)

    #print(class_weights)

if __name__=="__main__":
    data_dir = "/n/holyscratch01/protopapas_lab/Everyone/eghitmangilkes/data/imagenome/chest-imagenome/silver_dataset/feature_extraction_final/"
    output_dir = "/n/holyscratch01/protopapas_lab/Everyone/eghitmangilkes/data/imagenome/chest-imagenome/silver_dataset/class_counts/"
    train_set = "/n/holyscratch01/protopapas_lab/Everyone/eghitmangilkes/data/imagenome/chest-imagenome/silver_dataset/coco/updated/new_train.csv"
    filename = "train_class_counts.csv"
    train_ids = pd.read_csv(train_set)
    #print(train_ids)
    get_class_weights(data_dir, train_ids['image_id'], output_dir, filename)