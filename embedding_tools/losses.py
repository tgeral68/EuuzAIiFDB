import torch 

from function_tools import poincare_function as pf
from torch.nn import functional as tf

class SGALoss(object):
    @staticmethod
    def O1(x, y, distance=None):
        if(distance is None):
            distance = pf.distance
        return tf.logsigmoid(-((distance(x, y))**2))

    @staticmethod
    def O2(x, y, z, distance=None):
        if(distance is None):
            distance = pf.distance
        y_reshape = y.unsqueeze(2).expand_as(z)
        return SGALoss.O1(x, y, distance=distance) + tf.logsigmoid((distance(y_reshape,z))**2).sum(-1)


class SGDLoss(object):
    @staticmethod
    def O1(x, y, distance=None):
        if(distance is None):
            distance = pf.distance
        return -SGALoss.O1(x, y, distance=distance)

    @staticmethod
    def O2(x, y, z, distance=None):
        if(distance is None):
            distance = pf.distance
        return -SGALoss.O2(x, y, z, distance=distance)
