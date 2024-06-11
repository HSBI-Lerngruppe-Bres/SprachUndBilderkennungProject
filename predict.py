from ultralytics.models.yolo.classify import ClassificationPredictor


args = dict(model='yolov8n-cls.pt')
predictor = ClassificationPredictor(overrides=args)
predictor.predict_cli()
