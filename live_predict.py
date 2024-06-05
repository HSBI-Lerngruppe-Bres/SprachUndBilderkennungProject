from ultralytics import YOLO
import cv2
# Load your trained YOLO classification model
model = YOLO('runs/classify/train6/weights/best.pt')

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

    # Convert the frame to the format expected by the model
    input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # input_frame = cv2.resize(frame_gray, (model.img_size, model.img_size))

    # Run the frame through the YOLO model for classification
    results = model(input_frame)

    # Assuming the model returns a list of results with class names and confidence scores
    for result in results:
        # Convert probabilities to numpy array if needed
        probs = result.probs.cpu().numpy()

        # Get the index of the class with the highest probability
        class_idx = probs.top1
        print(model.names[class_idx])
        # Get the class name and confidence
        # label = result.names[class_idx]
        # confidence = probs[class_idx]

        # print(f"Class: {label}, Confidence: {confidence:.2f}")

    # Display the frame
    cv2.imshow('Live Stream', input_frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
