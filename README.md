# Multimodal Graph Representation Learning for Chest X-Rays

### Dependencies
* python 3.5+
* pytorch 1.0+
* torchvision
* numpy
* pandas
* sklearn
* matplotlib
* tensorboardX
* tqdm

### Dataset
Requires MIMIC-CXR Dataset and Chest Imagenome dataset which can both be found on https://physionet.org

to generate the format for object detection from the chest imagenome scene graphs run:
```
python ./data/coco_format.py
```

To train Detectron 2 Faster R-CNN on the coco format dataset, run:
```
python ./detection/object_detection.py
```

To extract the Anatomical region features from the Faster R-CNN model run:
```
python ./data/feature_extraction.py
```

To get the report text from the scene graphs and extract the Anatomical region features from RadBERT run:
```
python ./data/get_report_text.py
python ./data/get_radbert_states.py
```

To construct the adjacency matrix run:
```
python ./data/adjacency_matrix.py
``` 

### Usage
To train a model using default batch size, learning:
```
python train_bimodal.py  
```

### References
+ [AnaXnet: Anatomy Aware Chest X-ray findings classification network](https://github.com/Nkechinyere-Agu/AnaXNet)
+ [Joint Modeling of Chest Radiographs and Radiology Reports for Pulmonary Edema Assessment](https://github.com/RayRuizhiLiao/joint_chestxray)

