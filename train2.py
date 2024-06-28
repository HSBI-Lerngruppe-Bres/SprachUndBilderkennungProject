from ultralytics import YOLO

model_sizes = ['yolov8n-cls.pt']
datas = ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']
epochs = 100
patience = 20

# Hyperparameters selected by tuning
hyperparameters = {
    'lr0': 0.01,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'warmup_momentum': 0.8,
    'box': 7.5,
    'cls': 0.5,
    'dfl': 1.5,
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 0.0,
    'translate': 0.1,
    'scale': 0.5,
    'shear': 0.0,
    'perspective': 0.0,
    'flipud': 0.0,
    'fliplr': 0.5,
    'bgr': 0.0,
    'mosaic': 1.0,
    'mixup': 0.0,
    'copy_paste': 0.0
}

# Loop through each model size and data fold
for model_name in model_sizes:
    for data in datas:
        model = YOLO(model_name)
        args = {
            'data': data,
            'epochs': epochs,
            'patience': patience,
            'optimizer': "auto",
            'dropout': 0.2,
            ** hyperparameters
        }

        results = model.train(**args)
