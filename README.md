
# PromptDet: Expand Your Detector  Vocabulary with Uncurated Images
[Paper]() &nbsp; &nbsp; [Website](https://fcjian.github.io/promptdet)

## Introduction

The goal of this work is to establish a scalable pipeline for expanding an object detector towards novel/unseen categories, using *zero manual annotations*. To achieve that, we make the following four contributions: (i) in pursuit of generalisation, we propose a two-stage open-vocabulary object detector that categorises each box proposal by a classifier generated from the text encoder of a pre-trained visual-language model; (ii) To pair the visual latent space (from RPN box proposal) with that of the pre-trained text encoder, we propose the idea of *regional prompt learning* to optimise a couple of learnable prompt vectors, converting the textual embedding space to fit those visually object-centric images; (iii) To scale up the learning procedure towards detecting a wider spectrum of objects, we exploit the available online resource, iteratively updating the prompts, and later self-training the proposed detector with pseudo labels generated on a large corpus of noisy, uncurated web images. The self-trained detector, termed as **PromptDet**, significantly improves the detection performance on categories for which manual annotations are unavailable or hard to obtain, *e.g.* rare categories. Finally, (iv) to validate the necessity of our proposed components, we conduct extensive experiments on the challenging LVIS and MS-COCO dataset, showing superior performance over existing approaches with *fewer additional training images* and *zero manual annotations* whatsoever.

## Training framework
![method overview](resources/promptdet.png)

## Prerequisites

- MMDetection version 2.16.0.

- Please see [get_started.md](docs/get_started.md) for installation and the basic usage of MMDetection.

## Inference

```python
./tools/dist_test.sh configs/promptdet/promptdet_mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1.py work_dirs/promptdet_mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1.pth 4 --eval bbox segm
```

## Train
To be updated.

## Models

For your convenience, we provide the following trained models (TOOD). All models are trained with 16 images in a mini-batch.

Model | Epochs | Scale Jitter | Input Size | AP<sub>novel | AP<c>c | AP<sub>f | AP | Config | Download
--- |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
PromptDet_R_50_FPN_1x | 12 | 640~800  | 800x800 | 19.0 | 18.5 | 25.8 | 21.4 | [config](configs/promptdet/promptdet_mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1.py) | [google](https://drive.google.com/file/d/1M7ccIsfQKA5pEtgMlRSadokLu_cFKO4B/view?usp=sharing) / [baidu](https://pan.baidu.com/s/1rjAwcX2rq5xTm7_9AdWR2Q)
PromptDet_R_50_FPN_6x | 72 | 100~1280 | 800x800 | 21.4 | 23.3 | 29.3 | 25.3 | [config](configs/promptdet/promptdet_mask_rcnn_r50_fpn_sample1e-3_mstrain_6x_lvis_v1.py) | [google](https://drive.google.com/file/d/1G3Waqs3Xh7h1bfwcUfek91S1JKRCTAdV/view?usp=sharing) / [baidu](https://pan.baidu.com/s/1E_Lsxj4GXhe7iPL6feVa5Q)

[0] *All results are obtained with a single model and without any test time data augmentation such as multi-scale, flipping and etc..* \
[1] *Refer to more details in config files in `config/promptdet/`.* \
[2] *Extraction code of baidu netdisk: promptdet.*


## Acknowledgement

Thanks MMDetection team for the wonderful open source project!


## Citation

If you find Prompt useful in your research, please consider citing:

```

```


