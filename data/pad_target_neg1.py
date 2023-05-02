import json, os
from tqdm import tqdm
import numpy as np

if __name__=="__main__":
    diseaselist = ['lung opacity', 'pleural effusion', 'atelectasis', 'enlarged cardiac silhouette',
        'pulmonary edema/hazy opacity', 'pneumothorax', 'consolidation', 'fluid overload/heart failure', 'pneumonia']

    feature_dir = '/n/holyscratch01/protopapas_lab/Everyone/eghitmangilkes/data/imagenome/chest-imagenome/silver_dataset/feature_extraction/'
    neg1_feature_dir = '/n/holyscratch01/protopapas_lab/Everyone/eghitmangilkes/data/imagenome/chest-imagenome/silver_dataset/feature_extraction_neg1/'
    feature_files = os.listdir(feature_dir)
    for filename in feature_files:
        with open(os.path.join(feature_dir, filename), 'r') as f:
            img_feats = json.load(f)
        if np.array(img_feats['target']).shape[0] < 18:
            feat_sum = np.array(img_feats['features']).sum(axis=1)
            
            # missing_objs = df['features'].apply(lambda x: 1 if sum(x)==0 else 0).values
            missing_objs_idx = np.where(feat_sum==0)[0]
            t = np.array(img_feats['target'])
            for i in missing_objs_idx:
                if t.shape[0] == 0:
                    t = (np.ones(len(diseaselist)) * -1).reshape(1,-1)
                elif i == 0:
                    t = np.vstack([np.ones(len(diseaselist)) * -1, t])
                elif i == 17:
                    t = np.vstack([t, np.ones(len(diseaselist)) * -1])
                else:
                    t = np.vstack([t[:i], np.ones(len(diseaselist)) * -1, t[i:]])
        #     print(i)
            img_feats['target'] = t.tolist()
        with open(os.path.join(neg1_feature_dir, filename), 'w', encoding='utf-8') as f:
                json.dump(img_feats, f, ensure_ascii=False, indent=4)
    

