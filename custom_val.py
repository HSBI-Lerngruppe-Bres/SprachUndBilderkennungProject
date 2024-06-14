from ultralytics.models.yolo.classify import ClassificationValidator

args = dict(model="runs/classify/train5/weights/best.pt", data="V1")
validator = ClassificationValidator(args=args)
validator()
