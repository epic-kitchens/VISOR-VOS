# VISOR-VOS


This repository contains the codes to train STM on [VISOR](https://epic-kitchens.github.io/VISOR/) dataset [Space-Time Memory Networks (STM)](https://openaccess.thecvf.com/content_ICCV_2019/html/Oh_Video_Object_Segmentation_Using_Space-Time_Memory_Networks_ICCV_2019_paper.html)


# Requirements
- Python 3.9.7
- Numpy 1.20.3
- Pillow 8.4.0
- Opencv-python 4.5.5
- Imgaug 0.4.0
- Scipy 1.7.1
- Tqdm 4.62.3
- Pandas 1.3.4



#### Dataset Structure
```

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
![3x5_images-1](https://user-images.githubusercontent.com/24276671/192190134-eeb6492d-3e70-4363-8bdf-1f5ec5887fe4.png)

