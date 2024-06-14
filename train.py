from ultralytics.models.yolo.classify import ClassificationTrainer

model_sizes = ['yolov8n-cls.pt', 'yolov8s-cls.pt', 'yolov8m-cls.pt']
data = 'V1'
epochs = 300

for model in model_sizes:
    args = {
        'model': model,
        'data': data,
        'epochs': epochs,
    }
    trainer = ClassificationTrainer(overrides=args)
    trainer.train()
    print(f"Training complete for model {model}")
