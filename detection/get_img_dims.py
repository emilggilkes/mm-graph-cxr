import json, os
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd

image_root = '/n/holyscratch01/protopapas_lab/Everyone/eghitmangilkes/data/mimic/physionet.org/files/mimic-cxr-jpg/2.0.0/files/'
img_dir = os.listdir(image_root)

img_ids, heights, widths = [], [], []
for img_name in tqdm(img_dir[:188555]):
    img = cv2.imread(image_root+img_name, 0)
    height, width = img.shape[:2]
    heights.append(height)
    widths.append(width)
    img_ids.append(img_name.split('.')[0])

df = pd.DataFrame({'image_id': img_ids, 'height': heights, 'width': widths})
df.to_csv('cxr-img-dims.csv', index=False)
