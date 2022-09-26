# VISOR-VOS


This repository contains the codes to train STM on [VISOR](https://epic-kitchens.github.io/VISOR/) dataset [Space-Time Memory Networks (STM)](https://openaccess.thecvf.com/content_ICCV_2019/html/Oh_Video_Object_Segmentation_Using_Space-Time_Memory_Networks_ICCV_2019_paper.html)


# Requirements
- python 3.9.7
- Numpy 1.20.3
- Pillow 8.4.0
- opencv-python 4.5.5
- imgaug 0.4.0
- scipy 1.7.1
- tqdm 4.62.3
- pandas 1.3.4



#### Dataset Structure
```
 |- data
      |- VISOR
          |- JPEGImages
          |- Annotations
          |- ImageSets
             |- train.txt
             |- val.txt
             |- val_unseen.txt

      |- MS-COCO
          |- train2017
          |- annotations
              |- instances_train2017.json
```

