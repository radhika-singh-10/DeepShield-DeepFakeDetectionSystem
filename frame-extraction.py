import cv2
import os

def extract_faces(input_folder, output_folder, cascade_path):
    # Load the Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Check if output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through all images in the input folder
    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)

        # Check if the file is an image
        if image_name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
            # Read the image
            img = cv2.imread(image_path)

            # Convert image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Save each detected face as a separate file
            for i, (x, y, w, h) in enumerate(faces):
                face = img[y:y+h, x:x+w]
                output_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}_face_{i+1}.jpg")
                cv2.imwrite(output_path, face)

if __name__ == "__main__":
    # Define the input folder containing images
    real_uncleaned_train_input_folder = "/home/rsingh57/images-test/train-base/real"
    fake_uncleaned_train_input_folder = "/home/rsingh57/images-test/train-base/fake"
    real_uncleaned_val_input_folder = "/home/rsingh57/images-test/val-base/real"
    fake_uncleaned_train_input_folder = "/home/rsingh57/images-test/val-base/fake"

    # Define the output folder to save extracted faces
    real_cleaned_train_input_folder = "/home/rsingh57/images-test/train-base/aug_real"
    fake_cleaned_train_input_folder = "/home/rsingh57/images-test/train-base/aug_fake"
    real_cleaned_val_input_folder = "/home/rsingh57/images-test/val-base/aug_real"
    fake_cleaned_train_input_folder = "/home/rsingh57/images-test/val-base/aug_fake"

    # Path to Haar Cascade XML file
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    # Run the face extraction
    extract_faces(real_uncleaned_train_input_folder, real_cleaned_train_input_folder, cascade_path)
    print(f"Faces extracted and saved to '{real_cleaned_train_input_folder}'")

    extract_faces(fake_uncleaned_train_input_folder, fake_cleaned_train_input_folder, cascade_path)
    print(f"Faces extracted and saved to '{fake_cleaned_train_input_folder}'")

    extract_faces(real_uncleaned_val_input_folder, real_cleaned_val_input_folder , cascade_path)
    print(f"Faces extracted and saved to '{real_cleaned_val_input_folder}'")

    extract_faces(fake_uncleaned_train_input_folder, fake_cleaned_train_input_folder, cascade_path)
    print(f"Faces extracted and saved to '{fake_cleaned_train_input_folder}'")
   





