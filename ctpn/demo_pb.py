from __future__ import print_function

import glob
import os
import shutil
import sys

import cv2
import numpy as np
import tensorflow as tf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lib.fast_rcnn.config import cfg, cfg_from_file
from lib.fast_rcnn.test import _get_blobs
from lib.rpn_msr.proposal_layer_tf import proposal_layer
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg
from ctpn.tf_runtime import configure_tensorflow_runtime


def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale is not None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f


def draw_boxes(img, image_name, boxes, scale):
    base_name = os.path.basename(image_name)
    stem = os.path.splitext(base_name)[0]

    os.makedirs("data/results", exist_ok=True)
    with open(os.path.join("data", "results", "res_{}.txt".format(stem)), "w") as f:
        for box in boxes:
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                continue
            if box[8] >= 0.9:
                color = (0, 255, 0)
            elif box[8] >= 0.8:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
            cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)

            min_x = min(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            min_y = min(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
            max_x = max(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
            max_y = max(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
            f.write(",".join([str(min_x), str(min_y), str(max_x), str(max_y)]) + "\r\n")

    img = cv2.resize(img, None, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join("data", "results", base_name), img)


if __name__ == "__main__":
    configure_tensorflow_runtime()

    if os.path.exists("data/results/"):
        shutil.rmtree("data/results/")
    os.makedirs("data/results/")

    cfg_from_file("ctpn/text.yml")

    model_path = "data/ctpn_saved_model"
    if not os.path.isdir(model_path):
        raise RuntimeError(
            "SavedModel not found at {}. Run `python ctpn/generate_pb.py` first.".format(model_path)
        )

    loaded = tf.saved_model.load(model_path)
    infer = loaded.signatures["serving_default"]

    im_names = glob.glob(os.path.join(cfg.DATA_DIR, "demo", "*.png")) + glob.glob(
        os.path.join(cfg.DATA_DIR, "demo", "*.jpg")
    )

    for im_name in im_names:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(("Demo for {:s}".format(im_name)))
        img = cv2.imread(im_name)
        img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
        blobs, im_scales = _get_blobs(img, None)
        im_blob = blobs["data"]
        blobs["im_info"] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

        preds = infer(
            images=tf.convert_to_tensor(blobs["data"], dtype=tf.float32),
            im_info=tf.convert_to_tensor(blobs["im_info"], dtype=tf.float32),
        )
        cls_prob = preds["rpn_cls_prob_reshape"].numpy()
        box_pred = preds["rpn_bbox_pred"].numpy()

        rois, _ = proposal_layer(cls_prob, box_pred, blobs["im_info"], "TEST", anchor_scales=cfg.ANCHOR_SCALES)
        scores = rois[:, 0]
        boxes = rois[:, 1:5] / im_scales[0]
        textdetector = TextDetector()
        boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
        draw_boxes(img, im_name, boxes, scale)
