# Repository for the Project of IACV course at PoliMI

## Installation

```
git clone https://github.com/proceduralia/neural_enhance
cd neural_enhance
conda create --name myenv --file requirements.txt
source activate myenv
```

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
