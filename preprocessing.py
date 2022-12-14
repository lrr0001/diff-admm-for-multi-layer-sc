import tensorflow
import tensorflow.math
from tensorflow.keras.layers import Layer
from tensorflow import math
import tensorflow.nn
import numpy


class Local_contrast_normalization(Layer):

  def __init__(self,localSz,thrshld,**kwargs):
    self.localSz = localSz
    self.thrshld = thrshld
    super(Local_contrast_normalization, self).__init__(**kwargs)

  def build(self, input_shape):
    super(Local_contrast_normalization, self).build(input_shape)

  def call(self, x):
    sigmah = self.localSz/6.0
    gaussWeights = numpy.zeros((self.localSz,self.localSz,1,1),numpy.float32)
    for ii in range(self.localSz):
      for jj in range(self.localSz):
        gaussWeights[ii,jj,0,0] = numpy.exp(-((ii - self.localSz/2.0)**2 + (jj - self.localSz/2.0)**2)/(2.0*sigmah))
    gaussWeights = gaussWeights/numpy.sum(gaussWeights)
    localMean = tensorflow.nn.conv2d(input=x,filter=tensorflow.convert_to_tensor(gaussWeights),strides=None,padding="SAME")
    demeanedX = x - localMean
    normalizationFactor = tensorflow.math.sqrt(tensorflow.nn.conv2d(input=tensorflow.math.square(demeanedX),filter=tensorflow.convert_to_tensor(gaussWeights),strides=None,padding="SAME"))
    return tensorflow.where(normalizationFactor>self.thrshld,demeanedX/normalizationFactor,demeanedX)

  def compute_output_shape(self,input_shape):
    return input_shape  

