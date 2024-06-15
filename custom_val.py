from ultralytics.models.yolo.classify import ClassificationValidator


models = [
    "runs/classify/train45/weights/best.pt",
    "runs/classify/train46/weights/best.pt",
    "runs/classify/train47/weights/best.pt",
    "runs/classify/train48/weights/best.pt",
    "runs/classify/train49/weights/best.pt",
]

"""datasets = [
    "fold_reduced_1",
    "fold_reduced_2",
    "fold_reduced_3",
    "fold_reduced_4",
    "fold_reduced_5",
]"""

for model, data in zip(models, datasets):

    args = dict(model=model, data=data)
    validator = ClassificationValidator(args=args)
    metrics = validator()
    print(metrics)
    print(f"stats: {validator.get_stats()}")
    validator.print_results()

    metrics = validator.finalize_metrics()

    print(f"Metrics: {metrics}")
