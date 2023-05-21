import tensorflow as tf
import numpy as np
import cv2

# Load the Teachable Machine model
model = tf.keras.models.load_model('keras_model.h5')

# Load the class labels
with open('labels.txt', 'r') as f:
    class_labels = f.read().splitlines()


# Initialize the camera
video = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame from the camera
    ret, frame = video.read()

    # Preprocess the input image
    image = cv2.resize(frame, (224, 224))
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)

    # Make predictions
    predictions = model.predict(image)
    predicted_label_index = np.argmax(predictions[0])
    predicted_label = class_labels[predicted_label_index]

    # Display the predicted label on the frame
    cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Frame", frame)
    print(predicted_label)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the windows
video.release()
cv2.destroyAllWindows()



# Load and preprocess the input image
image = tf.keras.preprocessing.image.load_img('path_to_your_image.jpg', target_size=(224, 224))
image = tf.keras.preprocessing.image.img_to_array(image)
image = np.expand_dims(image, axis=0)
image = image / 255.0  # Normalize the image

# Make predictions
predictions = model.predict(image)
predicted_label_index = np.argmax(predictions[0])
predicted_label = class_labels[predicted_label_index]

# Print the predicted label
print("Predicted Label:", predicted_label)
