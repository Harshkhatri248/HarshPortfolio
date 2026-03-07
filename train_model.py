import cv2
import os
import numpy as np

def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    # Path to your face and the folder for 'other' faces
    data_folders = {
        "dataset/user": 1,    # Your ID
        "dataset/others": 2   # Everyone else
    }
    
    faces = []
    labels = []

    for folder, label_id in data_folders.items():
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created {folder}. Add some images there!")
            continue

        for file in os.listdir(folder):
            if file.endswith((".jpg", ".png")):
                img_path = os.path.join(folder, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                faces.append(img)
                labels.append(label_id)

    if len(faces) > 0:
        recognizer.train(faces, np.array(labels))
        recognizer.save("trainer.yml")
        print("✅ Success: trainer.yml updated with 'User' and 'Others'.")
    else:
        print("❌ Error: No images found to train.")

if __name__ == "__main__":
    train_model()