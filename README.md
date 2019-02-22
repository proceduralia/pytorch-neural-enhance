# Repository for the Project of IACV course at PoliMI

## Installation

```
git clone https://github.com/proceduralia/neural_enhance
cd neural_enhance
```

To train a model without conditioning run 

```
python main.py --model_name unet --loss l1nima
```
To see all options run:

```
python main.py -h
```

To train a model without condition run 

```
python condition_main.py --model_name unet --loss l1nima
```
To see all options run:

```
python condition_main.py -h
```
