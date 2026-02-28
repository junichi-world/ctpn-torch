from .ctpn_keras import CTPNModel


class VGGnet_train(CTPNModel):
    def __init__(self, trainable=True):
        super().__init__(name="VGGnet_train")
        self.trainable = trainable
