from __future__ import print_function

import os
import shutil
import sys

import tensorflow as tf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from lib.fast_rcnn.config import cfg, cfg_from_file
from lib.networks.factory import get_network
from ctpn.tf_runtime import configure_tensorflow_runtime


def find_latest_checkpoint(checkpoint_path):
    candidates = []
    if os.path.isabs(checkpoint_path):
        candidates.append(checkpoint_path)
    else:
        candidates.extend(
            [
                checkpoint_path,
                os.path.join(PROJECT_ROOT, checkpoint_path),
                os.path.join(PROJECT_ROOT, "ctpn", checkpoint_path),
            ]
        )

    seen = set()
    for candidate in candidates:
        norm = os.path.normpath(candidate)
        if norm in seen:
            continue
        seen.add(norm)
        latest = tf.train.latest_checkpoint(norm)
        if latest is not None:
            return latest, norm
    return None, checkpoint_path


class CTPNInferenceModule(tf.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.float32, name="images"),
            tf.TensorSpec(shape=[None, 3], dtype=tf.float32, name="im_info"),
        ]
    )
    def __call__(self, images, im_info):
        outputs = self.model(images, training=False)
        return {
            "rpn_cls_prob_reshape": outputs["rpn_cls_prob_reshape"],
            "rpn_bbox_pred": outputs["rpn_bbox_pred"],
        }


if __name__ == "__main__":
    configure_tensorflow_runtime()

    cfg_from_file("ctpn/text.yml")
    net = get_network("VGGnet_test")
    _ = net(tf.zeros([1, 64, 64, 3], dtype=tf.float32), training=False)

    ckpt = tf.train.Checkpoint(model=net)
    latest, resolved_ckpt_dir = find_latest_checkpoint(cfg.TEST.checkpoints_path)
    if latest is None:
        raise RuntimeError("No checkpoint found under {}".format(cfg.TEST.checkpoints_path))
    print("Restoring from {} (resolved from {})...".format(latest, resolved_ckpt_dir), end=" ")
    ckpt.restore(latest).expect_partial()
    print("done")

    export_dir = "data/ctpn_saved_model"
    if os.path.isdir(export_dir):
        shutil.rmtree(export_dir)

    module = CTPNInferenceModule(net)
    tf.saved_model.save(module, export_dir)
    print("SavedModel exported to {}".format(export_dir))
