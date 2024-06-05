import cv2
from ultralytics import YOLO
# from ultralytics.yolo.utils import render_result

# Load your trained YOLO classification model
model = YOLO('path_to_your_trained_model.pt')

# Open a connection to the webcam (or other video source)
cap = cv2.VideoCapture(0)  # 0 is the default camera

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Run the frame through the YOLO model for classification
    results = model(frame)

    # Render the results on the frame
    model.render_result(results, frame)

    # Display the frame
    cv2.imshow('Live Stream', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
