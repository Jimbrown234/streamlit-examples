from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
# import torchvision.transforms.functional as transform

import io
import os
from PIL import Image
import streamlit as st
import torch
import wget


import torchvision.transforms as transforms
  
transform = transforms.Compose([
    transforms.PILToTensor()
])

def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test: Object Detection')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        # return uploaded_file
        return transform(Image.open(io.BytesIO(image_data)))
    
    else:
        return None



def load_and_predict(img1): 
    
    # Step 1: Initialize model with the best available weights

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
    model.eval()

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = [preprocess(img1)]

    # Step 4: Use the model and visualize the prediction
    prediction = model(batch)[0]
    labels = [weights.meta["categories"][i] for i in prediction["labels"]]
    box = draw_bounding_boxes(img1, boxes=prediction["boxes"],
                              labels=labels,
                              colors="red",
                              width=4, font_size=30)
    im = to_pil_image(box.detach())
    # im.show()
    st.image(im)

def main():
    st.title('Pretrained Object Detection model demo using Faster R-CNN')
    
    img1 = (load_image())
    result = st.button('Run on image')
    if result:
        st.write('Detecting objects ...')
        load_and_predict(img1)

if __name__ == '__main__':
    main()