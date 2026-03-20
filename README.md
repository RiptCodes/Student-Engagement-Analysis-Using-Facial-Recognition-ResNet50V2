# DAiSEE Engagement Detection

## Setup
```
pip install -r requirements.txt
```

## File Structure
```
daisee_project/
    config.py        - all paths and hyperparameters
    preprocessing.py - face detection, data processing
    dataset.py       - DataGenerator
    model.py         - ResNet50V2 model
    prepare_data.py  - run first, processes videos into .npy
    train.py         - training loop
    evaluate.py      - evaluation, confusion matrix, confidence
    requirements.txt
```

## How To Run

**1. Edit config.py** — set your DATASET_PATH and PROJECT_PATH

**2. Process the dataset** (only need to do this once)
```
python prepare_data.py
```

**3. Train**
```
python train.py
```

**4. Evaluate**
```
python evaluate.py
```

## Check GPU is working
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```


## load up the python environment 
```
source daisee-env/bin/activate
```