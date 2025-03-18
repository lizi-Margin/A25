import os
import cv2
from third_party.DEYOLO.ultralytics import YOLO
from global_config import GlobalConfig as cfg


def train(model, dataset_yaml):
    # Train the model
    train_results = model.train(
        data=dataset_yaml,  # path to dataset YAML
        epochs=100,  # number of training epochs
        imgsz=640,  # training image size
        device=cfg.device,

        # hsv_h=0.2,
        # hsv_s=0.2
    )

    # Evaluate model performance on the validation set
    metrics = model.val()

    # print(train_results)
    # print(metrics)


if __name__ == '__main__':
    dataset_name = 'deyolo'
    # model_path = 'yolov8n.pt'
    model_path = 'deyolo_models/best.pt'

    root_dir = cfg.root_dir
    model = YOLO(f"{root_dir}/third_party/DEYOLO/ultralytics/models/v8/DEYOLO.yaml").load(model_path)
    dataset = f'{root_dir}/datasets/{dataset_name}/data.yaml'
    train(model, dataset)
