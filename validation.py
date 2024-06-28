from ultralytics import YOLO

# Load your trained model
model = YOLO('runs/classify/train196/weights/best.pt')

# Validate the model on the validation dataset
results = model.val(data='datasets/fold_4')
print(results)
