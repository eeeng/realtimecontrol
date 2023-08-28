import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

#from super_gradients.training import models
from ultralytics import YOLO
from mrcnn.visualize import display_instances
from mrcnn.config import Config
from mrcnn import model as modellib, utils

model=YOLO()


class CustomConfig(Config):
    """Configuration for training on the custom  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "object"

    # We use a GPU with 24GB memory, which can fit 4 images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 11  


    # Number of training steps per epoch
    STEPS_PER_EPOCH = 12

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    
if __name__ == "__main__":
    leads = ['Lead I', 'Lead II', 'Lead III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    dx_dict = {
        '426783006': 'SNR', # Normal sinus rhythm
        '164889003': 'AF', # Atrial fibrillation
        '270492004': 'IAVB', # First-degree atrioventricular block
        '164909002': 'LBBB', # Left bundle branch block
        '713427006': 'RBBB', # Complete right bundle branch block
        '59118001': 'RBBB', # Right bundle branch block
        '284470004': 'PAC', # Premature atrial contraction
        '63593006': 'PAC', # Supraventricular premature beats
        '164884008': 'PVC', # Ventricular ectopics
        '429622005': 'STD', # ST-segment depression
        '164931005': 'STE', # ST-segment elevation
    }
    classes = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']

############################################################
#  Dataset
############################################################

class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):
      
      
        # Add classes. We have only one class to add.
        self.add_class("object", 1, "Lead I")
        self.add_class("object", 2, "Lead II")
        self.add_class("object", 3, "Lead III")
        self.add_class("object", 4, "V1")
        self.add_class("object", 5, "V2")
        self.add_class("object", 6, "V3")
        self.add_class("object", 7, "V4")
        self.add_class("object", 8, "V5")
        self.add_class("object", 9, "V6")
        self.add_class("object", 10, "aVR")
        self.add_class("object", 11, "aVL")
        self.add_class("object", 12, "aVF")
        self.add_class("object", 13, "RA")
        self.add_class("object", 14, "RL")
        self.add_class("object", 15, "LL")
        self.add_class("object", 16, "LA")   
       

with open("ECGleads.txt","r") as f:
    classes=[line.strip()for line in f.readlines()]
    
cap=cv2.VideoCapture(0)

print("ECGleads.txt")

while(1):
    _, frame = cap.read()
