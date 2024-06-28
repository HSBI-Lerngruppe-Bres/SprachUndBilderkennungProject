from ultralytics.models.yolo.classify import ClassificationTrainer

# List of model configurations and data folds
model_sizes = ['yolov8n-cls.pt']

datas = ['fold_reduced_2_1', 'fold_reduced_2_2',
         'fold_reduced_2_3', 'fold_reduced_2_4', 'fold_reduced_2_5']

"""model_sizes = ['yolov8n-cls.pt', "yolov8s-cls.pt", "yolov8m-cls.pt"]
datas = ["fold_1"]"""

# Training parameters
epochs = 100
patience = 5

# Loop through each model size and data fold
for model in model_sizes:
    for data in datas:
        args = {
            'model': model,
            'data': data,
            'epochs': epochs,
            'patience': patience,
        }

        # Initialize the trainer with the specified arguments
        trainer = ClassificationTrainer(overrides=args)

        # Train the model with the found optimal learning rate
        trainer.train()
        trainer.final_eval()
