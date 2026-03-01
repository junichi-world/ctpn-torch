import cv2
import numpy as np
import tensorflow as tf

from .config import cfg
from lib.utils.blob import im_list_to_blob


def _get_image_blob(im):
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im_resized = cv2.resize(
            im_orig, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR
        )
        im_scale_factors.append(im_scale)
        processed_ims.append(im_resized)

    blob = im_list_to_blob(processed_ims)
    return blob, np.array(im_scale_factors)


def _get_blobs(im, rois):
    blobs = {"data": None, "rois": None}
    blobs["data"], im_scale_factors = _get_image_blob(im)
    return blobs, im_scale_factors


def test_ctpn(net, im, boxes=None):
    blobs, im_scales = _get_blobs(im, boxes)
    if cfg.TEST.HAS_RPN:
        im_blob = blobs["data"]
        blobs["im_info"] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)
    else:
        raise ValueError("Only RPN mode is supported")

    rois, _, _ = net.predict_rois(
        tf.convert_to_tensor(blobs["data"], dtype=tf.float32),
        tf.convert_to_tensor(blobs["im_info"], dtype=tf.float32),
        cfg_key="TEST",
    )
    rois = rois.numpy()

    scores = rois[:, 0]
    assert len(im_scales) == 1, "Only single-image batch implemented"
    boxes = rois[:, 1:5] / im_scales[0]
    return scores, boxes
