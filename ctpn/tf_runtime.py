import os

import tensorflow as tf


def configure_tensorflow_runtime():
    """Configure TensorFlow to use GPU when available, otherwise CPU."""
    # Keep memory growth enabled to avoid grabbing all VRAM at startup.
    try:
        gpus = tf.config.list_physical_devices("GPU")
    except Exception as exc:
        print("TensorFlow GPU probe failed: {}".format(exc))
        return []

    if not gpus:
        print("TensorFlow GPU not available; using CPU.")
        print("Hint: in Linux/WSL install `tensorflow[and-cuda]` in the `ctpn` env.")
        return []

    visible_index = os.environ.get("CTPN_GPU_INDEX")
    if visible_index is not None:
        try:
            idx = int(visible_index)
            if 0 <= idx < len(gpus):
                tf.config.set_visible_devices(gpus[idx], "GPU")
                gpus = [gpus[idx]]
        except Exception as exc:
            print("Ignoring invalid CTPN_GPU_INDEX={}: {}".format(visible_index, exc))

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as exc:
            print("Could not enable memory growth for {}: {}".format(gpu, exc))

    print("TensorFlow using GPU(s): {}".format(", ".join(d.name for d in gpus)))
    return gpus

