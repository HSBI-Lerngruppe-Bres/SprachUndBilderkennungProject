from ultralytics.models.yolo.classify import ClassificationValidator

args = dict(model="runs/classify/train3/weights/best.pt", data="Max")
validator = ClassificationValidator(args=args)
validator()
