# CACViT(AAAI 24)
This repository is the official implementation of our AAAI 2024 Paper [Vision Transformer Off-the-Shelf: A Surprising Baseline for Few-Shot Class-Agnostic Counting](https://ojs.aaai.org/index.php/AAAI/article/view/28396)

## Installation
Our code has been tested on Python 3.8.18 and PyTorch 1.8.1+cu111. Please follow the official instructions to setup your environment. See other required packages in `requirements.txt`.

## Data Preparation
We train and evaluate our methods on FSC-147 dataset. Please follow the FSC-147 official repository to download and unzip the dataset.

* [FSC147](https://github.com/cvlab-stonybrook/LearningToCountEverything)

## Inference
We have included our pretrained model here [link](https://pan.baidu.com/s/1VUF3WBXbUvGsSQj9lG7tLQ?pwd=4nca). Then you can run following command to conduct the inference on the FSC-147 dataset. 

```
python test.py
```

## Training
We use the same pretrained MAE model as CounTR, please download the pretrained MAE weight as [CounTR](https://github.com/Verg-Avesta/CounTR). Then you can run the following command to conduct the traininng on the FSC-147 dataset.

```
python train_val.py
```

## Citation
If you find this work or code useful for your research, please cite:

```
@inproceedings{wang2024vision,
  title={Vision transformer off-the-shelf: A surprising baseline for few-shot class-agnostic counting},
  author={Wang, Zhicheng and Xiao, Liwen and Cao, Zhiguo and Lu, Hao},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2024}
}
```

