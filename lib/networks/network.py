from .ctpn_keras import CTPNModel


class Network(CTPNModel):
    """Backward-compatible alias for the legacy graph-based Network class."""

    def __init__(self, *args, **kwargs):
        super().__init__(name="Network")
