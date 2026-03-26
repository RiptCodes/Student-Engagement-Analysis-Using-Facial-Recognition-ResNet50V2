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

## Setting up camera for the WSLinux

##### 1. Open PowerShell As Administrator
##### 2. Paste this:
```
usbipd list 
```
##### 3. Find your camera in the device list and take the list number "X.X"

##### 4. Paste this: 
```
usbipd bind --busid X-X
usbipd attach --wsl --busid X-X
```
##### 5. Device is now linked to WSL. Go back to command line and type (only if you receive errors):
```
sudo chmod 666 /dev/video0 /dev/video1
```
##### 6. Enter password for Ubuntu and then Run Demo.py. If you want to change the model access config.py and change `MODEL_PATH`
```
python3 demo.py
```