# DSproject3
The purpose of this project is to understand object detection methods using YOLO (You Only Look Once) models with images of various fruits. Below is docoumentation for reproducability

## Software and Platform
This project used Python programming thorugh Jupyter Notebook on Windows platform.

Required installations:
```
pip install ultralytics

import torch
import zipfile
import os
import pandas as pd
from PIL import Image
import numpy as np
import yaml
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
```
## Documentation
- LISCENCE.MD: proper citation for repository
- SCRIPTS folder: source code for project
- DATA folder: initial data (imported test/training images and fruit labels) and final cleaned data
- OUTPUT: figures and tables generated from scripts
- REFERENCES: sources used throughout project

## Reproducing Results
SCRIPTS: The script provided is the final script to clean the data, perform an EDA, training/testing YOLO model, and find precision of the model.

CLEAN DATA & EDA: read in original test/train data and labels (source: https://data.mendeley.com/datasets/5prc54r4rt), combine images and labels 
for ease of analysis. 
- OUTPUT: EDA graphs, merged_train_df, merged_test_df.

ANALYSIS: train YOLO model on images, and test model to detect certain fruits. Use mean average precision (mAP) to find precision of model.
- OUTPUT: mAP values
