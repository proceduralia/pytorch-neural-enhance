# Repository for the Project of IACV course at PoliMI

## Installation

```
git clone https://github.com/proceduralia/neural_enhance
cd neural_enhance
conda create --name myenv --file requirements.txt
source activate myenv
```

To download the dataset run:
```
python scrape_fivek.py --base_dir path/to/data
```

## Training
To train a model without conditioning run 

```
python main.py --model_type unet --loss l1nima --data_path path/to/data
```
To see all options run:

```
python main.py -h
```

To train a model without condition run 

```
python conditioned_main.py --model_type unet --loss l1nima --data_path path/to/data
```
To see all options run:

```
python conditioned_main.py -h
```
## Evaluation
To evaluate a model without condition run 
```
python evaluate.py --model_type unet --image_path path/to/image --final_dir path/to/model_folder
```
To see all options run:

```
python evaluate.py -h
```


N.B inside the path/to/model_folder there should be a file "*.pth" who's name strats with the model type
