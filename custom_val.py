from ultralytics.models.yolo.classify import ClassificationValidator


models = [
    "runs/classify/train30/weights/last.pt",
    "runs/classify/train31/weights/last.pt",
    "runs/classify/train32/weights/last.pt",
    "runs/classify/train33/weights/last.pt",
    "runs/classify/train34/weights/last.pt",
]

datasets = [
    "fold_1",
    "fold_2",
    "fold_3",
    "fold_4",
    "fold_5",
]

for model, data in zip(models, datasets):

    args = dict(model=model, data=data)
    validator = ClassificationValidator(args=args)
    validator()
    print(f"stats: {validator.get_stats()}")
    validator.print_results()

    metrics = validator.finalize_metrics()

    print(f"Metrics: {metrics}")
