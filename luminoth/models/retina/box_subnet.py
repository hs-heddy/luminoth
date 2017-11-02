import tensorflow as tf

from luminoth.models.retina.subnet import Subnet


class BoxSubnet(Subnet):
    def __init__(self, config, num_anchors, name='box_subnet'):
        num_final_channels = num_anchors * 4
        final_bias = tf.constant_initializer(0)
        super(BoxSubnet, self).__init__(
            config, num_final_channels, final_bias=final_bias, name=name
        )
        self._config = config
