import os
import re
import cv2
import numpy as np
import supervision as sv
from typing import Union, List
from ultralytics import YOLO
from UTIL.colorful import *
from siri_utils.sleeper import Sleeper
from global_config import GlobalConfig as cfg
from pre.extract_number import extract_number


model_path = './best.pt'
model = YOLO(model=model_path, task='detect')
tracker = sv.ByteTrack()

def _predict(frame_or_batch: Union[np.ndarray, List[np.ndarray]]):
    if isinstance(frame_or_batch, np.ndarray):
        batch = [frame_or_batch]
    elif isinstance(frame_or_batch, list):
        assert isinstance(frame_or_batch[0], np.ndarray)
        batch = frame_or_batch
    else:
        assert False

    assert len(batch[0].shape) == 3
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    # frame = pad(frame, to_sz_wh=cfg.sz_wh)
    # frame = cv2.resize(frame, cfg.sz_wh)
    # if cfg.manual_preprocess: 
    #     batch = preprocess(batch)

    results = model.predict(
        batch,
        cfg=f"{cfg.root_dir}/yolo_model/game.yaml",
        imgsz=tuple(reversed(cfg.sz_wh)),
        stream=True,   # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! stream set
        conf=cfg.conf_threshold,
        iou=0.5,
        device=cfg.device,
        half=cfg.half,
        max_det=20,
        agnostic_nms=False,
        augment=False,
        vid_stride=False,
        visualize=False,
        verbose=False,
        show_boxes=False,
        show_labels=False,
        show_conf=False,
        save=False,
        show=False,
        # batch=1
    )

    return results

def test(imgdir, output_video="output.mp4", fps=30):
    assert os.path.exists(imgdir)

    all_img = sorted(os.listdir(imgdir), key=extract_number)
    first_frame = cv2.imread(f"{imgdir}/{all_img[0]}")
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    try:
        for i, img in enumerate(all_img):
            print绿(f"\r testing frame {i}: {img}", end='')

            frame = cv2.imread(f"{imgdir}/{img}")
            assert len(frame.shape) == 3
            results = _predict(frame)
            result = next(results)

            detections = sv.Detections.from_ultralytics(result)
            detections = tracker.update_with_detections(detections)

            annotator = sv.BoxAnnotator()
            frame = annotator.annotate(scene=frame, detections=detections)

            video_writer.write(frame)
        #     cv2.imshow("Detection", frame)
        #     cv2.waitKey(0)
    finally:
        # cv2.destroyAllWindows()
        video_writer.release()       

if __name__ == '__main__':
    # test('./datasets/wl_test/', fps=24)
    test('./datasets/ir_test/', fps=24)


