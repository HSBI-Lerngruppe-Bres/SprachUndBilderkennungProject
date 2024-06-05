from ultralytics.models.yolo.classify import ClassificationTrainer

args = dict(model='yolov8n-cls.pt', data='V1', epochs=3)

trainer = ClassificationTrainer(overrides=args)
trainer.train()
