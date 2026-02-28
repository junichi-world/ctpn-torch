from __future__ import print_function

import os
import re
import shutil
import sys
import glob

import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lib.fast_rcnn.config import cfg, cfg_from_file
from lib.networks.factory import get_network


class CTPNInferenceModule(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images, im_info):
        outputs = self.model(images, training=False)
        return outputs["rpn_cls_prob_reshape"], outputs["rpn_bbox_pred"]


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


if __name__ == "__main__":
    cfg_from_file("ctpn/text.yml")
    net = get_network("VGGnet_test")

    device = torch.device(f"cuda:{cfg.GPU_ID}" if torch.cuda.is_available() else "cpu")
    latest = _find_latest_checkpoint(cfg.TEST.checkpoints_path)
    if latest is None:
        raise RuntimeError("No checkpoint found under {}".format(cfg.TEST.checkpoints_path))

    print("Restoring from {}...".format(latest), end=" ")
    payload = torch.load(latest, map_location=device)
    net.load_state_dict(payload["model"])
    net.to(device)
    net.eval()
    print("done")

    export_path = "data/ctpn_torchscript.pt"
    if os.path.isfile(export_path):
        os.remove(export_path)

    module = CTPNInferenceModule(net).to(device)
    module.eval()
    scripted = torch.jit.script(module)
    scripted.save(export_path)
    print("TorchScript exported to {}".format(export_path))
