# BGMix in PyTorch

Implementation of "Background-Mixed Augmentation for Weakly Supervised Change Detection" in PyTorch

## Requirements
- python 3.9
- pytorch 1.9.1
- opencv-python 4.5.5.64
- torchvision 0.10.1
- pillow 9.1.1



## Train
You can use the following command to train：
> python train.py 

- `train.py`: the entry point for training.
- `models/CG.py`: defines the architecture of the Generator models and Discriminator models.
- `options.py`: creates option lists using `argparse` package.
- `datasets.py`: process the dataset before passing to the network.
- `models/vgg16.py`: defines the Classifier.
- `models/models.py`: defines the model.
- `optimizer.py`: defines the optimizetion.
- `loss.py`: defines the loss functions.



### Test
You can use the following command to test：
> python test.py 

### Pretrained Classifier
Because of perception similarity loss, you need to use the following command to train the Classifier firstly.
> python train_Classifier.py



## Datasets
- `train_data`: The data for training.
  - `AICD`:  Aerial image change detection.
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




