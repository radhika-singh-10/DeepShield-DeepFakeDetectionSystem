import numpy as np
import pandas as pd
import os
import random
from PIL import Image
import matplotlib.pyplot as plt

def main():
    # Specify the folder path
    folder_path = 'final_data/train-base/aug_fake'

    # Get all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    # Check if there are at least 5 images in the folder
    if len(image_files) < 5:
        print("Not enough images in the folder to select 5.")
    else:
        # Randomly select 5 images
        selected_images = random.sample(image_files, 5)
        
        # Display the selected images
        plt.figure(figsize=(15, 5))
        for i, img_file in enumerate(selected_images):
            img_path = os.path.join(folder_path, img_file)
            img = Image.open(img_path)
            plt.subplot(1, 5, i + 1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(img_file)
        plt.tight_layout()
        plt.show()



if __name__ == '__main__':
    main()