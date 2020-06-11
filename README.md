# Pytorch Neural Enhance
![ ](https://proceduralia.github.io/assets/can.png)

This repository contains the code for the experiments presented in the technical report [An empirical evaluation of convolutional neural networks for image
enhancement](https://proceduralia.github.io/assets/IACV_Project.pdf).

The repository contains the code for evaluating CAN32 and UNet models, together with a number of different loss functions, on the task of supervised image enhancement, imitating experts from the MIT-Adobe FiveK dataset.

The models can also be conditioned, by *conditional batch normalization*, on the categorical features contained in the dataset.

![ ](https://proceduralia.github.io/assets/flower.png)

In addition, the script *learn_transform.py* performs training for learning Contrast Limited Adaptive Histogram Equalization (CLAHE) on the CIFAR10 dataset, using different architectures.

---

Written in collaboration with [ennnas](https://github.com/ennnas) for the Computer Vision MSc course at Politecnico di Milano.

## Installation

```
git clone https://github.com/proceduralia/neural_enhance
cd neural_enhance
conda create --name myenv --file requirements.txt
source activate myenv
```

To download the MIT-Adobe FiveK dataset run:

```
python scrape_fivek.py --base_dir path/to/data
```

## Training
To train a model without using categorical features as additional input run: 

```
python main.py --model_type unet --loss l1nima --data_path path/to/data
```

To train a model using categorical features as additional input run:

```
python conditioned_main.py --model_type unet --loss l1nima --data_path path/to/data
```

## Evaluation
To evaluate a model (without conditions) run:
```
python evaluations.py --model_type unet --image_path path/to/image --final_dir path/to/model_folder
```

## Citation
If you found this repository or the report useful for your research work, you can cite them:
```
@misc{pytorchenhance,
  author = {Nasca, Ennio and D'Oro, Pierluca},
  title = {An empirical evaluation of convolutional neural networks for image enhancement},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/proceduralia/pytorch-neural-enhance}},
}
```
