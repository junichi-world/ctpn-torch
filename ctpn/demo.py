from __future__ import print_function

import glob
import os
import re
import shutil
import sys

import cv2
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lib.fast_rcnn.config import cfg, cfg_from_file
from lib.fast_rcnn.test import test_ctpn
from lib.networks.factory import get_network
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg
from lib.utils.timer import Timer


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


def _find_latest_checkpoint(directory):
    if not os.path.isdir(directory):
        return None

    files = glob.glob(os.path.join(directory, "ctpn_iter_*.pth"))
    if not files:
        return None

    def _step(path):
        m = re.search(r"ctpn_iter_(\d+)\.pth$", os.path.basename(path))
        return int(m.group(1)) if m else -1

    files.sort(key=_step)
    return files[-1]


def ctpn(net, image_name):
    timer = Timer()
    timer.tic()
    img = cv2.imread(image_name)
    img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    scores, boxes = test_ctpn(net, img)
    textdetector = TextDetector()
    boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    draw_boxes(img, image_name, boxes, scale)
    timer.toc()
    print(("Detection took {:.3f}s for {:d} object proposals").format(timer.total_time, boxes.shape[0]))


if __name__ == "__main__":
    if os.path.exists("data/results/"):
        shutil.rmtree("data/results/")
    os.makedirs("data/results")

    cfg_from_file("ctpn/text.yml")
    net = get_network("VGGnet_test")
    device = torch.device(f"cuda:{cfg.GPU_ID}" if torch.cuda.is_available() else "cpu")
    net.to(device)

    print(("Loading network {:s}... ".format("VGGnet_test")), end=" ")
    latest = _find_latest_checkpoint(cfg.TEST.checkpoints_path)
    if latest is None:
        raise RuntimeError("No checkpoint found under {}".format(cfg.TEST.checkpoints_path))
    print("Restoring from {}...".format(latest), end=" ")
    payload = torch.load(latest, map_location=device)
    net.load_state_dict(payload["model"])
    net.eval()
    print("done")

    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for _ in range(2):
        test_ctpn(net, im)

    im_names = glob.glob(os.path.join(cfg.DATA_DIR, "demo", "*.png")) + glob.glob(
        os.path.join(cfg.DATA_DIR, "demo", "*.jpg")
    )
    for im_name in im_names:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(("Demo for {:s}".format(im_name)))
        ctpn(net, im_name)
