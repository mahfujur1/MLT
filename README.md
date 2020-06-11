# Mixture Label Transport

This repository contains the implementation of the NeurIPS-2020 submission [Bridging the Gap Between Supervised and Unsupervised Performance on Person Re-identification via Mixture Label Transport](https://nips.cc/).
**Please note that** it is a *pre-released* repository for the anonymous review process, and the *official* repository will be released upon the paper published.

Code is coming soon.


## Requirements

### Installation

```shell
git clone https://github.com/MLT-reid/MLT
cd MLT

```

### Prepare Datasets

```shell
cd examples && mkdir data
```
Download the person datasets [DukeMTMC-reID](https://arxiv.org/abs/1609.01775), [Market-1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view), [MSMT17](https://arxiv.org/abs/1711.08565), Then unzip them under the directory like
```
SpCL/examples/data
├── dukemtmc
│   └── DukeMTMC-reID
├── market1501
│   └── Market-1501-v15.09.15
├── msmt17
    └── MSMT17_V1

```

You can create the soft link to the dataset:
```shell
ln -s /path-to-data ./data
```

ImageNet-pretrained models for **ResNet-50** will be automatically downloaded in the python script.


## Training

We utilize 4 GPUs for training. **Note that**


### Unsupervised Domain Adaptation
To train the model(s) in the paper, run this command:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/spcl_train_uda.py -ds $SOURCE_DATASET -dt $TARGET_DATASET --logs-dir $PATH_LOGS
```

*Example #1:* DukeMTMC-reID -> Market-1501
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/spcl_train_uda.py -ds dukemtmc -dt market1501 --logs-dir logs/spcl_uda/duke2market_resnet50
```
*Example #2:* DukeMTMC-reID -> MSMT17
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/spcl_train_uda.py -ds dukemtmc -dt msmt17 --iters 800 --logs-dir logs/spcl_uda/duke2msmt_resnet50
```
*Example #3:* VehicleID -> VeRi
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python examples/spcl_train_uda.py -ds vehicleid -dt veri --iters 800 --height 224 --width 224 --logs-dir logs/spcl_uda/vehicleid2veri_resnet50
```


## Evaluation

We utilize 1 GTX-1080TI GPU for testing. **Note that**

+ use `--width 128 --height 256` (default) for person datasets, and `--height 224 --width 224` for vehicle datasets;
+ use `--dsbn` for domain adaptive models, and add `--test-source` if you want to test on the source domain;
+ use `-a resnet50` (default) for the backbone of ResNet-50, and `-a resnet_ibn50a` for the backbone of IBN-ResNet.

### Unsupervised Domain Adaptation

To evaluate the model on the target-domain dataset, run:

```shell
CUDA_VISIBLE_DEVICES=0 python examples/test.py --dsbn -d $DATASET --resume $PATH_MODEL
```

To evaluate the model on the source-domain dataset, run:

```shell
CUDA_VISIBLE_DEVICES=0 python examples/test.py --dsbn --test-source -d $DATASET --resume $PATH_MODEL
```

*Example #1:* DukeMTMC-reID -> Market-1501
```shell
# test on the target domain
CUDA_VISIBLE_DEVICES=0 python examples/test.py --dsbn -d market1501 --resume logs/spcl_uda/duke2market_resnet50/model_best.pth.tar
# test on the source domain
CUDA_VISIBLE_DEVICES=0 python examples/test.py --dsbn --test-source -d dukemtmc --resume logs/spcl_uda/duke2market_resnet50/model_best.pth.tar
```

### Unsupervised Learning
To evaluate the model, run:
```shell
CUDA_VISIBLE_DEVICES=0 python examples/test.py -d $DATASET --resume $PATH
```

*Example #1:* DukeMTMC-reID
```shell
CUDA_VISIBLE_DEVICES=0 python examples/test.py -d dukemtmc --resume logs/spcl_usl/duke_resnet50/model_best.pth.tar
```

## Trained Models

![framework](figs/results.png)

You can download the above models in the paper from [Google Drive](https://drive.google.com/open?id=19vYA4EfInuH4ZKg0HeBRmDmgK1KLdivz).
