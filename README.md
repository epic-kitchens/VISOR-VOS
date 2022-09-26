# VISOR-VOS


This repository contains the codes to train STM on [VISOR](https://epic-kitchens.github.io/VISOR/) dataset [Space-Time Memory Networks (STM)](https://openaccess.thecvf.com/content_ICCV_2019/html/Oh_Video_Object_Segmentation_Using_Space-Time_Memory_Networks_ICCV_2019_paper.html)

<br>

## Performance on VISOR

| backbone |  training stage | training dataset | J&F | J |  F  | weights |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| resnet-50 |  stage 1 | MS-COCO | 56.9 | 55.5 | 58.2 | [`link`](https://www.dropbox.com/s/bsy577kflurboav/coco_res50.pth?dl=0) |
| resnet-50 | stage 2 | MS-COCO -> VISOR | 75.8 | 73.6 | 78.0 | [`link`](https://www.dropbox.com/s/6vkkr6vbx7ybku3/coco_lr_fix_skip_0_1_release_resnet50_400000_32_399999.pth?dl=0) |


# Requirements
- Python 3.9.7
- Numpy 1.20.3
- Pillow 8.4.0
- Opencv-python 4.5.5
- Imgaug 0.4.0
- Scipy 1.7.1
- Tqdm 4.62.3
- Pandas 1.3.4
- Torchvision 0.12.0

## Datasets

#### [MS-COCO](https://cocodataset.org/#home)
MS-COCO instance segmentation dataset is used to generate synthitic video out of 3 frames to train STM. This could be helpful as a pretraining stage before doing the main training on VISOR. <br>

![image](https://user-images.githubusercontent.com/19390123/115352832-62fb7d00-a1ea-11eb-9fbe-1f84bf74905d.png)


#### [VISOR](https://epic-kitchens.github.io/VISOR/)
After pretrain on MS-COCO, we fine-tune on VISOR dataset by sample 3 frames from a sequence in each training iteration. To visualize VISOR dataset, you can check [VISOR-VIS](https://github.com/epic-kitchens/VISOR-VIS)
![00230](https://user-images.githubusercontent.com/24276671/192192037-bec3f981-0cc5-405d-85bc-610e883d0466.jpg)


#### Dataset Structure
To run the training or evaluation scripts, the dataset format should be as follows (following [DAVIS](https://davischallenge.org/) format):
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


## Training

#### Stage 1
To pretrain on MS-COCO, you can run the following command.
```
python train_coco.py -Dvisor "path to visor" -Dcoco "path to coco" -backbone "[resnet50,resnet18]" -save "path to checkpoints"
#e.g.
python train_coco.py -Dvisor ../data/Davis/ -Dcoco ../data/Ms-COCO/ -backbone resnet50 -save ../coco_weights/
```

#### Stage 2
Main traning on VISOR, to get the best performance, you should resume from the MS-COCO pretrained model in Stage 1.
```
python train_stm_baseline.py -Dvisor "path to visor" -backbone "[resnet50,resnet18]" -save "path to checkpoints" -resume "path to coco pretrained weights"
#e.g. 
train_stm_baseline.py -Dvisor ../data/VISOR/ -backbone resnet50 -save ../visor_weights/ -resume ../coco_weights/coco_res50.pth
```

## Evaluation
Evaluating on VISOR based on DAVIS evaluation codes, we adjusted the codes to include the last frame of the sequence in our scores 
```
python eval.py -g "gpu id" -s "set" -y "year" -D "path to visor" -p "path to weights" -backbone "[resnet50,resnet18,resnest101]"
#e.g.
python eval.py -g 0 -s val -y 17 -D ../data/VISOR -p ../visor_weights/coco_lr_fix_skip_0_1_release_resnet50_400000_32_399999.pth -backbone resnet50
```



## Acknowledgement

When use this repo, any of our models or dataset, you need to cite the VISOR paper

## Citing VISOR
```
@inproceedings{VISOR2022,
  title = {EPIC-KITCHENS VISOR Benchmark: VIdeo Segmentations and Object Relations},
  author = {Darkhalil, Ahmad and Shan, Dandan and Zhu, Bin and Ma, Jian and Kar, Amlan and Higgins, Richard and Fidler, Sanja and Fouhey, David and Damen, Dima},
  booktitle = {Proceedings of the Neural Information Processing Systems (NeurIPS) Track on Datasets and Benchmarks},
  year = {2022}
}
```

We use the code in the original STM implementation from [official STM repository](https://github.com/seoungwugoh/STM) and the implementation from [STM training repository](https://github.com/haochenheheda/Training-Code-of-STM). Using this code, you also need to cite STM

## Citing STM
```
@inproceedings{oh2019video,
  title={Video object segmentation using space-time memory networks},
  author={Oh, Seoung Wug and Lee, Joon-Young and Xu, Ning and Kim, Seon Joo},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={9226--9235},
  year={2019}
}
```
