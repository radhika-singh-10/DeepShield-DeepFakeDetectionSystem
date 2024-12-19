# CSE-573-DeepFake-Image-Detection

# Project Overview - Comparative Analysis of Deep Fake Detection System for
Ensuring Integrity in Images

The detection of Deepfake content on social media is a well-researched problem, 
with significant contributions made by architectures like ForensicTransfer and Improved VGG-CNN, 
Resnet-InceptionNet, MobileNet, etc. These methods, however, exhibit limitations such as high computational costs, 
poor generalization, low-quality deepfake detection, and a primary focus on extracting only spatial features using 
CNNs that eventually decrease the efficiency of the system. Current benchmark models also rely on non-diverse datasets
 and perform poorly in real-world scenarios.

This project focuses on conducting a comparative analysis of existing and a new deep fake detection system that 
leverages the state of the art. The new deepfake detection vision model is an ensemble deep learning model that is 
trained on the best parameters and is capable of classifying images as either real (0) or fake (1), along with 
providing a probability score to indicate the confidence level of the prediction. The application is designed to 
assist in various domains, including content moderation platforms, educational platforms for spreading awareness 
about deepfakes, social media to combat misinformation, and research platforms for strengthening cybersecurity models. 
The inputs of the system are the model accepts pre-processed image files (resized, normalized, or grayscale) and 
outputs binary classification labels (0-real, 1-fake).

This project proposes a novel solution that overcomes these challenges by performing tasks such as incorporating 
both spatial and temporal features for robust detection. While developing the system I have taken performed research 
analysis on multiple state-of-the-art models to learn the best and the worst characteristics that helped in developing 
the newer lightweight deep stack architecture for better performance and scalability. 
In thew beginning, I have conducted custom pre-processing of the images for efficient data handling, 
including resizing, normalization, and facial landmark detection by utilizing benchmark datasets like 
FaceForensics++, Celeb-DF, and Deepfake Detection Challenge to ensure diversity in training and evaluation.
Through these contributions, this project bridges gaps in existing solutions and delivers a resource-efficient, 
scalable, and accurate deepfake detection model suited for real-world applications.




# Setup:

- Install required modules via `requirement.txt` file
- The command for requirement.txt - pip install -r requirements.txt
- Load the data from the url in this format in the code folder - https://buffalo.app.box.com/folder/297332721000
- The data will have the below format
- Make run.sh excutable by running chmod +x run.sh 
- Run the shell script run.sh. The command is sh run.sh or bash run.sh


# Note:

The shell file will run the frame-extraction.py, state-of-the-art-model.py, and visualization-generation.py in the ascending order

Feel free to change the order or comment any line for testing and playing around!


# Data Link

- Final Cleaned Data 
    - Train Data
        - Augumented Real Data
        - Augumented Fake Data

    - Validation Data
        - Augumented Real Data
        - Augumented Fake Data

# Requirements:

- python >= 3.6
- requirements.txt
