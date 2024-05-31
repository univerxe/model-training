from keras.models import load_model 
import cv2  # Install opencv-python
import numpy as np

np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = ["tired", "study", "sleep"]

camera = cv2.VideoCapture(0)

# Define window size
window_width = 640
window_height = 480

while True:
    ret, image = camera.read()
    resized_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Resize the display window
    cv2.namedWindow("Webcam Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Webcam Image", window_width, window_height)

    # Make the image a numpy array and reshape it to the model's input shape
    image_np = np.asarray(resized_image, dtype=np.float32).reshape(1, 224, 224, 3)
    image_np = (image_np / 127.5) - 1

    # Predict using the model
    prediction = model.predict(image_np)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Prepare the text to be displayed
    text = f"{class_name}: {np.round(confidence_score * 100, 2)}%"

    # Put the text on the image
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Webcam Image", image)

    # Print prediction and confidence score
    print(f"Class: {class_name}, Confidence Score: {np.round(confidence_score * 100, 2)}%")

    keyboard_input = cv2.waitKey(1)
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
