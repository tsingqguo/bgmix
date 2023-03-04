# BGMix in PyTorch

Implementation of "Background-Mixed Augmentation for Weakly Supervised Change Detection" in PyTorch.

## &#x1F4D6;Pipeline of the BGMix
<!-- ![image1](./images/bgmix.jpg) -->
<img src="./images/bgmix.jpg" alt="drawing" width="600"/>

<!-- ## &#x1F4D6;Visual comparison of CD results -->
<!-- ![image2](./images/bgmix4.jpg) -->
<!-- <img src="./images/bgmix4.jpg" alt="drawing" width="600"/> -->


## &#x1F4D6;Requirements

```
- python 3.9
- pytorch 1.9.1
- opencv-python 4.5.5.64
- torchvision 0.10.1
- pillow 9.1.1
```


## &#x1F4D6;Train
You can use the following commands to train and test：
> python train.py 
>
> python test.py

- `train.py`: the entry point for training.
- `models/CG.py`: defines the architecture of the Generator model and Discriminator models.
- `options.py`: creates option lists using the `argparse` package.
- `datasets.py`: process the dataset before passing it to the network.
- `models/vgg16.py`: defines the Classifier.
- `models/models.py`: defines the model.
- `optimizer.py`: defines the optimization.
- `loss.py`: defines the loss functions.



### &#x1F4D4;Pretrained Classifier
Because of the perceptual similarity loss, you need to train a Classifier to extract the semantic features.
> python train_Classifier.py



## &#x1F4D6;Dataset Preparation

### &#x1F4D4;Data structure
- `train_data`: The data for training.
  - `AICD`:  Aerial image change detection dataset.
    - `C`: Change images.
    - `UC`: Background images.
  - `BCD`: Building change detection dataset.
    - `C`: Change images.
    - `UC`: Background images.
- `test_data`: The data for testing.
  - `AICD`:  Aerial image change detection.
    - `C`: Change images.
  - `BCD`: Building change detection dataset.
    - `C`: Change images.

### &#x1F4D4;Data Download 
You can download the AICD dataset from [**The Aerial Imagery Change Detection (AICD) dataset**](https://computervisiononline.com/dataset/1105138664)

You can download the BCD dataset from the [**WHU Building change detection Dataset**](https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html)

Both cropped datasets can be downloaded [**here**](https://pan.baidu.com/s/1JvBcqOHhw8hJCnuJTKcDHw?pwd=bgmx ). Please cite their papers.

## &#x1F4D6;Visualization results
### &#x1F4D4;Examples of augmented image pairs
- AICD
<img src="./images/AICD_Aug.jpg" alt="drawing" width="1000"/>
- BCD
<img src="./images/BCD_Aug.jpg" alt="drawing" width="1000"/>

### &#x1F4D4;Examples of CD results
- AICD
<img src="./images/AICD.jpg" alt="drawing" width="1000"/>
- BCD
<img src="./images/BCD.jpg" alt="drawing" width="1000"/>

## :speech_balloon: Bibtex
If you find this repo useful for your research, please cite our paper:
```
@article{huang2023bgmix,
  title={Background-Mixed Augmentation for Weakly Supervised Change Detection},
  author={Huang, Rui and Wang, Ruofei and Guo, Qing and Wei, Jieda and Zhang, Yuxiang and Fan, Wei and Liu, Yang},
  journal={AAAI},
  year={2023}
}
```