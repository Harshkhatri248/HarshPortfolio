import cv2
import os

# Haar cascade path
cascade_path = r"C:\Users\HP\anaconda3\envs\faceenv\Library\etc\haarcascades\haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Create dataset folder if not exists
save_path = "dataset/user"
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

print("Press 'c' to capture face | Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if cv2.waitKey(1) & 0xFF == ord('c'):
            count += 1
            cv2.imwrite(f"{save_path}/face_{count}.jpg", face_img)
            print(f"Saved image {count}")

    cv2.imshow("Capture Faces", frame)

    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("Face capture completed")

