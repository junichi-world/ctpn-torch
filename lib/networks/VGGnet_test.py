from .ctpn_keras import CTPNModel


class VGGnet_test(CTPNModel):
    def __init__(self, trainable=True):
        super().__init__(name="VGGnet_test")
        self.trainable = trainable
