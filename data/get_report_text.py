import json, os
from tqdm import tqdm
import numpy as np
import pandas as pd

if __name__=="__main__":
    feature_dir = '/n/holyscratch01/protopapas_lab/Everyone/eghitmangilkes/data/imagenome/chest-imagenome/silver_dataset/feature_extraction_final/'
    scene_dir = '/n/holyscratch01/protopapas_lab/Everyone/eghitmangilkes/data/imagenome/chest-imagenome/silver_dataset/scene_graph/'
    feature_files = os.listdir(feature_dir)
    mm_feature_dir = '/n/holyscratch01/protopapas_lab/Everyone/eghitmangilkes/data/imagenome/chest-imagenome/silver_dataset/mm_feature_extraction_final_emptystr_list/'

    organs = ["right lung", "right apical zone", "right upper lung zone", "right mid lung zone", 
        "right lower lung zone", "right hilar structures", "right costophrenic angle", "left lung", "left apical zone",
        "left upper lung zone", "left mid lung zone", "left lower lung zone", "left hilar structures", 
        "left costophrenic angle", "mediastinum", "upper mediastinum", "cardiac silhouette", "trachea"]

    empty_counts = np.zeros(len(organs))
    for filename in tqdm(feature_files[:118922]):
        with open(os.path.join(feature_dir, filename), 'r') as f:
            img_feats = json.load(f)
        with open(os.path.join(scene_dir, filename[:-5] +'_SceneGraph.json'), 'r') as f:
            img_scene = json.load(f)
        report_text = [[""] for i in range(len(organs))]
        
        for i, obj in enumerate(organs):
            obj_id = img_feats['image_id'] + '_' + obj
            for attribute in img_scene['attributes']:
                if attribute['object_id'] == obj_id:
                    # report_text[i] = ' '.join(attribute['phrases'])
                    report_text[i] = attribute['phrases']
        img_feats['report_text'] = report_text
        empties = [i for i in range(len(report_text)) if report_text[i] == [""] ]
        # empties = np.where(np.array(report_text)==np.array([""]))[0]
        empty_counts[empties] += 1
        with open(os.path.join(mm_feature_dir, filename), 'w', encoding='utf-8') as f:
            json.dump(img_feats, f, ensure_ascii=False, indent=4)
    
    pd.DataFrame(data=empty_counts,columns=['num_empty']).to_csv(os.path.join(mm_feature_dir, "empty_counts.csv"), index=False)