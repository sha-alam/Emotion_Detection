import cv2
import numpy as np
from keras.models import model_from_json
import matplotlib.pyplot as plt

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json and create model
json_file = open('models.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("models.h5")
print("Loaded model from disk")

# List of image paths
# List of image paths
image_paths = [
    "test_image/1.jpg",
    "test_image/2.jpg",
    "test_image/3.jpg",
    "test_image/4.jpg",
    "test_image/5.jpg",
    "test_image/6.jpg",
    "test_image/7.jpg", 
    ]

# Process each image
for image_path in image_paths:
    # Load the image
    image = cv2.imread(image_path)

    # Check if the image loaded successfully
    if image is None:
        print(f"Error: Unable to load image from path: {image_path}")
        continue  #
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find faces in the image
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    faces = face_detector.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

    # String to store emotion predictions
    emotion_results = ""

    # Process each face in the image
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray = gray_image[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

        # Predict emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        max_index = int(np.argmax(emotion_prediction))
        emotion_label = emotion_dict[max_index]
        confidence = emotion_prediction[0][max_index]

        # Append emotion and confidence to the result string
        emotion_results += f"{emotion_label}: {confidence:.2f}\n"

    # Plot the image with emotion predictions
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Emotion Detection")
    plt.text(10, 10, emotion_results, color='white', backgroundcolor='black', fontsize=8, verticalalignment='top')
    plt.axis('off')
    plt.show()