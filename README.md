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

## Datasets

#### [MS-COCO](https://cocodataset.org/#home)
MS-COCO instance segmentation dataset is used to generate synthitic video out of 3 frames to train STM. This could be helpful as a pretraining stage before doing the main training on VISOR. <br>

![image](https://user-images.githubusercontent.com/19390123/115352832-62fb7d00-a1ea-11eb-9fbe-1f84bf74905d.png)


#### [VISOR](https://epic-kitchens.github.io/VISOR/)
After pretrain on MS-COCO, we fine-tune on VISOR dataset by sample 3 frames from a sequence in each training iteration. To visualize VISOR dataset, you can check [VISOR-VIS](https://github.com/epic-kitchens/VISOR-VIS)
![00230](https://user-images.githubusercontent.com/24276671/192192037-bec3f981-0cc5-405d-85bc-610e883d0466.jpg)


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

