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
from pre.dataloader import DEYOLO_Dataset
from torch.utils.data import DataLoader
from siri_utils.preprocess import embed_image_in_black_bg, resize_image_to_width

def combime_wl_ir(wl_frame, ir_frame):
    wl_height, wl_width = wl_frame.shape[:2]
    ir_height, ir_width = ir_frame.shape[:2]
    if wl_height > ir_height:
        ir_frame = embed_image_in_black_bg(ir_frame, wl_height)
    else:
        wl_frame = embed_image_in_black_bg(wl_frame, ir_height)
    combined_frame = np.hstack((wl_frame, ir_frame))
    MAX_WIDTH = 1200
    if combined_frame.shape[1] > MAX_WIDTH:
        combined_frame = resize_image_to_width(combined_frame, MAX_WIDTH)
    return combined_frame


model_path = './deyolo_models/best.pt'
model = YOLO(model_path)
tracker = sv.ByteTrack()

def _predict(frame_or_batch):
    # if isinstance(frame_or_batch, list):
    #     assert len(frame_or_batch[0]) == 2
    #     batch = frame_or_batch
    # else:
    #     assert False

    # assert len(batch[0][0].shape) == 3



    # assert isinstance(frame_or_batch[0], np.ndarray)
    # assert isinstance(frame_or_batch[1], np.ndarray)
    # assert len(frame_or_batch) == 2
    batch = frame_or_batch

    results = model.predict(
        batch,
        cfg=f"{cfg.root_dir}/yolo_model/game.yaml",
        imgsz=tuple(reversed(cfg.sz_wh)),
        # imgsz=640,
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

        # show_boxes=False,
        # show_labels=False,
        # show_conf=False,
        save=True,
        show=False,
        # batch=1
    )

    return results

def test(imgdir, output_video="output.mp4", fps=30):
    assert os.path.exists(imgdir)

    dataset = DEYOLO_Dataset(imgdir)

    first_frame = combime_wl_ir(*dataset[0])
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    try:
        for i, (wl_frame, ir_frame) in enumerate(dataset):
            print绿(f"\r testing frame {i}: {wl_frame.shape}, {ir_frame.shape}", end='')

            # wl_frame = 'datasets/deyolo_test/wl_wuxi_2_0004.jpg' 
            # ir_frame = 'datasets/deyolo_test/ir_wuxi_2_0004.jpg' 
            # results = _predict([[wl_frame, ir_frame,], [wl_frame, ir_frame,]])
            results = _predict([wl_frame, ir_frame,])
            if isinstance(results, list):
                result = results[0]
            else:
                result = next(results)
            # print(len(result))

            # img_with_boxes = result.plot()
            # cv2.imshow("YOLOv8 Detection", img_with_boxes)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # continue

            detections = sv.Detections.from_ultralytics(result)
            detections = tracker.update_with_detections(detections)

            annotator = sv.BoxAnnotator()
            wl_frame = annotator.annotate(scene=wl_frame, detections=detections)
            ir_frame = annotator.annotate(scene=ir_frame, detections=detections)

            video_writer.write(combime_wl_ir(wl_frame, ir_frame))
    finally:
        video_writer.release()       

if __name__ == '__main__':
    test('./datasets/deyolo_test/', output_video='output_deyolo.mp4', fps=24)


