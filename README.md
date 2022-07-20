
# PromptDet: Towards Open-vocabulary Detection using Uncurated Images (ECCV 2022)
[Paper](https://arxiv.org/abs/2203.16513) &nbsp; &nbsp; [Website](https://fcjian.github.io/promptdet)

## Introduction

The goal of this work is to establish a scalable pipeline for expanding an object detector towards novel/unseen categories, using *zero manual annotations*. To achieve that, we make the following four contributions: (i) in pursuit of generalisation, we propose a two-stage open-vocabulary object detector, where the class-agnostic object proposals are classified with a text encoder from pre-trained visual-language model; (ii) To pair the visual latent space (of RPN box proposals) with that of the pre-trained text encoder, we propose the idea of *regional prompt learning* to align the textual embedding space with regional visual object features; (iii) To scale up the learning procedure towards detecting a wider spectrum of objects, we exploit the available online resource via a novel self-training framework, which allows to train the proposed detector on a large corpus of noisy uncurated web images. Lastly, (iv) to evaluate our proposed detector, termed as **PromptDet**, we conduct extensive experiments on the challenging LVIS and MS-COCO dataset. PromptDet shows superior performance over existing approaches with *fewer additional training images* and *zero manual annotations whatsoever*.

## Training framework
![method overview](resources/promptdet.png)

**updates**
- July 20, 2022: add the code for LAION-novel and self-training
- March 28, 2022: initial release

## Prerequisites

- MMDetection version 2.16.0.

- Please see [get_started.md](docs/get_started.md) for installation and the basic usage of MMDetection.

## LAION-novel dataset
For your convenience, we provide the learned [prompt vectors](promptdet_resources/prompt_learner/lvis/model.pth.tar-6) and the [LAION-novel]() dataset to reproduct the results quickly.

And you also can learn the prompt vectors using the offline [RPL](https://github.com/fcjian/RPL), and generate the LAION-novel dataset using the [tools](tools/promptdet) of PromptDet as follows:
```python
# generate the category embeddings
python tools/promptdet/gen_category_embedding.py --model-file promptdet_resources/prompt_learner/lvis/model.pth.tar-6 --name-file promptdet_resources/lvis_category_and_description.txt --out-file promptdet_resources/lvis_category_embeddings.pt

# install the dependencies and retrival the LAION images
pip install faiss-cpu==1.7.2 img2dataset==1.12.0 fire==0.4.0 h5py==3.6.0
python tools/promptdet/retrieval_laion_image.py --indice-folder /mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/fengchengjian/backup/home/data/laion400m-64GB-index --metadata /mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/fengchengjian/backup/home/data/metadata.hdf5

# download the LAION images
python tools/promptdet/download_laion_image.py

# convert the LAION images to mmdetection format
python tools/promptdet/laion_lvis_novel.py --data-path data/laion_lvis/images --out-file data/laion_lvis/laion_train.json --base-ind-file promptdet_resources/lvis_base_inds.txt
```

## Inference

```python
# assume that you are under the root directory of this project,
# and you have activated your virtual environment if needed,
# and with LVIS v1.0 dataset in 'data/lvis_v1'.

./tools/dist_test.sh configs/promptdet/promptdet_r50_fpn_sample1e-3_mstrain_1x_lvis_v1_self_train.py work_dirs/promptdet_r50_fpn_sample1e-3_mstrain_1x_lvis_v1_self_train.pth 4 --eval bbox segm
```

## Train
```python
# download 'lvis_v1_train_seen.json' to 'data/lvis_v1/annotations'.

# train detector without self-training
./tools/dist_train.sh configs/promptdet/promptdet_r50_fpn_sample1e-3_mstrain_1x_lvis_v1.py 4

# train detector with self-training
./tools/dist_train.sh configs/promptdet/promptdet_r50_fpn_sample1e-3_mstrain_1x_lvis_v1_self_train.py 4 --resume-from work_dirs/promptdet_r50_fpn_sample1e-3_mstrain_1x_lvis_v1/epoch_6.pth
```
[0] *Annotation file of base categories: [lvis_v1_train_seen.json](https://drive.google.com/file/d/1dZQ5ytHgJPv4VgYOyjJerq4adc6GQkkd/view?usp=sharing).*

## Models

For your convenience, we provide the following trained models (PromptDet) with mask AP.

Model | RPL | Self-training | Epochs | Scale Jitter | Input Size | AP<sub>novel | AP<c>c | AP<sub>f | AP | Download
--- |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
Baseline |  | | 12 | 640~800  | 800x800 | 7.4 | 17.2 | 26.1 | 19.0 | [google]() / [baidu]()
[PromptDet_R_50_FPN_1x (w/o self-training)](configs/promptdet/promptdet_r50_fpn_sample1e-3_mstrain_1x_lvis_v1.py) | &check; | | 12 | 640~800  | 800x800 | 11.5 | 19.4 | 26.7 | 20.9 | [google](https://drive.google.com/file/d/1vsqhieOcR_s1dt0JNikQqB8OgsTYJNH-/view?usp=sharing) / [baidu]()
[PromptDet_R_50_FPN_1x](configs/promptdet/promptdet_r50_fpn_sample1e-3_mstrain_1x_lvis_v1_self_train.py) | &check; | &check; | 12 | 640~800  | 800x800 | 19.5 | 18.2 | 25.6 | 21.3 | [google](https://drive.google.com/file/d/1OkQbe_uM8i5DhXT82HMOYMBo7v0atYZD/view?usp=sharing) / [baidu]()

[0] *All results are obtained with a single model and without any test time data augmentation such as multi-scale, flipping and etc..* \
[1] *Refer to more details in config files in `config/promptdet/`.* \
[2] *Extraction code of baidu netdisk: promptdet.*

## Acknowledgement

Thanks [MMDetection](https://github.com/open-mmlab/mmdetection) team for the wonderful open source project!


## Citation

If you find PromptDet useful in your research, please consider citing:

```
@inproceedings{feng2022promptdet,
    title={PromptDet: Towards Open-vocabulary Detection using Uncurated Images},
    author={Feng, Chengjian and Zhong, Yujie and Jie, Zequn and Chu, Xiangxiang and Ren, Haibing and Wei, Xiaolin and Xie, Weidi and Ma, Lin},
    journal={Proceedings of the European Conference on Computer Vision},
    year={2022}
}
```


