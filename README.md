# Pose-Estimation-with-Message-Passing-Networks

This repository contains the code for my master's thesis Pose Estimation with Graph Neural Networks.
Follow-up work lead to his publication: [[Paper]](https://arxiv.org/abs/2110.05132)

## Main Results
### Results on COCO test-dev2017 with multi-scale test
| Method             | Backbone | Input size  |    AP | Ap .5 | AP .75 | AP (M) | AP (L) |
|--------------------|----------|-------------|-------|-------|--------|--------|--------|
| OpenPose\*         |    -     | -           | 61.8  | 84.9  |  67.5  |  57.1  |  68.2  | 
| Hourglass          | Hourglass  | 512       | 56.6  | 81.8  |  61.8  |  49.8  |  67.0  | 
| PersonLab          | ResNet-152  | 1401     | 66.5  | 88.0  |  72.6  |  62.4  |  72.3  |
| PifPaf             |    -     | -           | 66.7  | -     |  -     |  62.4  |  72.9  | 
| Bottom-up HRNet    | HRNet-w32  | 512       | 64.1  | 86.3  |  70.4  |  57.4  |  73.9  | 
| HigherHRNet        | HRNet-w48  | 640       | 70.5 | 89.3 | 77.2 | 66.6 | 75.8 | 74.9  | 
| **Ours**               | Higher HRNet-w48  | 640       | **71.0** | **89.5** | **77.3** | **67.6** | **76.2** |
### Results on CrowdPose test
| Method             |    AP | Ap .5 | AP .75 | AP (E) | AP (M) | AP (H) |
|--------------------|-------|-------|--------|--------|--------|--------|
| Mask-RCNN          | 57.2  | 83.5  | 60.3   | 69.4   | 57.9   | 45.8   |
| AlphaPose          | 61.0  | 81.3  | 66.0   | 71.2   | 61.4   | 51.1   |
| SPPE               | 66.0. | 84.2 | 71.5 | 75.5 | 66.3 | 57.4 |
| OpenPose           | - | - | - | 62.7 | 48.7 | 32.3 |
| HigherHRNet        | 65.9 | 86.4 | 70.6 | 73.3 | 66.5 | 57.9 |
| HigherHRNet+       | 67.6 | 87.4 | 72.6 | 75.8 | 68.1 | 58.9 |
| **Ours** | **69.0** | **88.3** | **73.3** | **77.3** | **69.8** | **60.4** |

*Note: + indicates using multi-scale test.*
## Start
### Installation
Most of the steps are similar to the steps described in [HigherHRNet](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
1. Install pytorch >= v1.4.0 following [official instruction](https://pytorch.org/).  
   - **Tested with pytorch v1.4.0**
2. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   # Install into global site-packages
   make install
   # Alternatively, if you do not have permissions or prefer
   # not to install the COCO API into global site-packages
   python3 setup.py install --user
   ```
   Note that instructions like # COCOAPI=/path/to/install/cocoapi indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (COCOAPI in this case) accordingly.
5. Install [CrowdPoseAPI](https://github.com/Jeff-sjtu/CrowdPose) exactly the same as COCOAPI.  
   - **There is a bug in the CrowdPoseAPI, please reverse https://github.com/Jeff-sjtu/CrowdPose/commit/785e70d269a554b2ba29daf137354103221f479e**
6. Init log(trained model, evaluation results, tensorboard log directory) directory:

   ```
   mkdir log
   mkdir PretrainedModels
   ```

   Your directory tree should look like this:

   ```
   ${POSE_ROOT}
   ├── data
   ├── experiments
   ├── log
   ├── PretrainedModels
   ├── src
   ├── README.md
   ├── requirements.txt
   └── ...
   ```

7. Download pretrained models from our [GoogleDrive](https://drive.google.com/drive/folders/1g08EYdqxnI3BudH2SAStLCFytjYUD9Yc?usp=sharing).Additionally, If you want to train our models from scratch you will need the pretrained models from [HigherHRNet](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
   ```
   PretrainedModels
   ├── 58_4.pth
   ├── 81_1_2.pth
   ├── pose_higher_hrnet_w32_512.pth
   └── ...

   ```
   
### Data preparation

**For COCO data**, please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation.
Download and extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        `-- images
            |-- train2017
            |   |-- 000000000009.jpg
            |   |-- 000000000025.jpg
            |   |-- 000000000030.jpg
            |   |-- ... 
            `-- val2017
                |-- 000000000139.jpg
                |-- 000000000285.jpg
                |-- 000000000632.jpg
                |-- ... 
```

**For CrowdPose data**, please download from [CrowdPose download](https://github.com/Jeff-sjtu/CrowdPose#dataset), Train/Val is needed for CrowdPose keypoints training and validation.
Download and extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- crowd_pose
    `-- |-- json
        |   |-- crowdpose_train.json
        |   |-- crowdpose_val.json
        |   |-- crowdpose_trainval.json (generated by tools/crowdpose_concat_train_val.py)
        |   `-- crowdpose_test.json
        `-- images
            |-- 100000.jpg
            |-- 100001.jpg
            |-- 100002.jpg
            |-- 100003.jpg
            |-- 100004.jpg
            |-- 100005.jpg
            |-- ... 
```
We follow previous methods and train our model on train and validation split (trainval split). Follow [HigherHRNet](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation) in order to create the trainval split.
After downloading data, run `python tools/crowdpose_concat_train_val.py` under `${POSE_ROOT}` to create trainval set.

### Training and Testing

#### Testing on COCO val2017 dataset and CrowdPose test dataset using our models ([GoogleDrive](https://drive.google.com/drive/folders/1g08EYdqxnI3BudH2SAStLCFytjYUD9Yc?usp=sharing))
 

For single-scale testing:

```
python valid.py --config hybrid_class_agnostic_end2end/model_58_4.yaml \
		     --out_file single_scale.txt \
         MODEL.PRETRAINED '../PretrainedModels/58_4.pth'
         DATASET.ROOT '../data/coco' \
         LOG_DIR '../log/58_4' \
		     MODEL.GC.POOL_KERNEL_SIZE 5 \
		     MODEL.GC.MASK_CROWDS False  \
		     MODEL.GC.CC_METHOD GAEC \
		     MODEL.MPN.NODE_THRESHOLD 0.1 \
		     MODEL.GC.GRAPH_TYPE knn \
		     TEST.SPLIT coco_17_full \
		     TEST.WITH_REFINE True \
		     TEST.FLIP_TEST True \
		     TEST.SCORING mean \
		     TEST.SCALE_FACTOR '[1.0]'
```

Multi-scale testing is also supported:

```
python valid.py --config hybrid_class_agnostic_end2end/model_58_4.yaml \
		     --out_file multi_scale.txt \
         MODEL.PRETRAINED '../PretrainedModels/58_4.pth' \
         DATASET.ROOT '../data/coco' \
         LOG_DIR '../log/58_4' \
		     MODEL.GC.POOL_KERNEL_SIZE 5 \
		     MODEL.GC.MASK_CROWDS False  \
		     MODEL.GC.CC_METHOD GAEC \
		     MODEL.MPN.NODE_THRESHOLD 0.1 \
		     MODEL.GC.GRAPH_TYPE knn \
		     TEST.SPLIT coco_17_full \
		     TEST.WITH_REFINE True \
		     TEST.FLIP_TEST True \
		     TEST.SCORING mean \
		     TEST.SCALE_FACTOR '[2.0, 1.0, 0.5]'
```
Multi-scale testing on CrowdPose test dataset:

```
python valid.py --config hybrid_class_agnostic_end2end_crowdpose/model_81_1_2.yaml \
		     --out_file multi_scale.txt \
         MODEL.PRETRAINED '../PretrainedModels/81_1_2.pth' \
         DATASET.ROOT '../data/crowd_pose' \
         LOG_DIR '../log/81_1_2' \
		     MODEL.GC.POOL_KERNEL_SIZE 5 \
		     MODEL.GC.MASK_CROWDS False  \
		     MODEL.GC.CC_METHOD GAEC \
		     MODEL.MPN.NODE_THRESHOLD 0.1 \
		     MODEL.GC.GRAPH_TYPE knn \
		     TEST.SPLIT crowd_pose_test \
		     TEST.WITH_REFINE True \
		     TEST.FLIP_TEST True \
		     TEST.SCORING mean \
		     TEST.SCALE_FACTOR '[2.0, 1.0, 0.5]'
```

#### Training on COCO train2017 dataset

```
python train.py hybrid_class_agnostic_end2end/model_58_4
```
This command will start the training using the model_58_4.yaml configuration file. Before executing this command please open the configuration file adapt the DATASET.ROOT variable such that it contains the new dataset root directory. In this case '${POSE_ROOT}/coco'.
#### Training on CrowdPose trainval dataset

```
python train.py hybrid_class_agnostic_end2end/model_81_1_2
```
This command will start the training using the model_81_1_2.yaml configuration file. Before executing this command please open the configuration file adapt the DATASET.ROOT variable such that it contains the new dataset root directory. In this case '${POSE_ROOT}/crowd_pose'.
