from keras import backend as K
from keras.layers import Layer
import tensorflow as tf
from tensorflow import math
import numpy as np
import keras.constraints

def get_effective_filter_size(kerSz,poolSz,nol):
  filterSz = 1
  for ii in range(nol-1,0,-1):
    filterSz = (filterSz + kerSz[ii] - 1)*poolSz[ii - 1]
  return filterSz + kerSz[0] - 1

def get_padding_size(inputSz,kerSz,poolSz,nol):
  effFilterSz = get_effective_filter_size(kerSz,poolSz,nol)
  paddingSz = 2*effFilterSz - 2
  currSz = inputSz + paddingSz - kerSz[0] + 1
  if currSz < 1:
    paddingSz = paddingSz + 1 - currSz
    currentSz = 1
  #endif
  for ii in range(1,nol):
    currRem = currSz % poolSz[ii - 1]
    if currRem > 0:
      currSz = currSz + poolSz[ii - 1] - currRem
      paddingSz = paddingSz + (poolSz[ii - 1] - currRem)*np.prod(poolSz[0:ii - 1])
    #endif
    currSz = int(currSz/poolSz[ii - 1]) - kerSz[ii] + 1
    if currSz < 1:
      paddingSz = paddingSz + (1 - currSz)*np.prod(poolSz[0:ii])
      currSz = 1
    #endif
  #endfor
  return (paddingSz,currSz)

class shrinkage_layer(Layer):

  def __init__(self,lambduh,**kwargs):
    self.lambduh = lambduh
    super(shrinkage_layer, self).__init__(**kwargs)

  def build(self, input_shape):
    super(shrinkage_layer, self).build(input_shape)

  def call(self, x):
    y = x - self.lambduh*tf.math.sign(x)
    return tf.where_v2(x <= -self.lambduh or x >= self.lambduh,y,0.)

  def compute_output_shape(self, input_shape):
    return input_shape



class Unpool_LS(Layer):

  def __init__(self, poolSz, **kwargs):
    # poolSz is purely used for assertions made in the build method.
    self.poolSz = poolSz
    super(Unpool_LS, self).__init__(**kwargs)

  def build(self, input_shape):
    inputShape = input_shape[0]
    pooledShape = input_shape[1]
    assert(inputShape[1]/float(self.poolSz[0]) == float(pooledShape[1]))
    assert(inputShape[2]/float(self.poolSz[1]) == float(pooledShape[2]))
    assert(inputShape[3] == pooledShape[3]) 
    super(Unpool_LS, self).build(input_shape)

  def call(self, x):
    # ***INPUTS***
    # x[0]: "noisy measurement" of signal pre-max_pooling
    # x[1]: "noisy measurement" of signal post-max_pooling
    #
    # ***OUTPUTS***
    # output: Least-squares estimate of signal pre-max_pooling
    # output2: Least-squares estimate of signal post-max_pooling
    #
    # ***CONCEPT**
    # For each block, do the following:
    #   0) Let S = empty
    #   1) Of the elements from the block not in set S, add the index of the maximum element to set S.
    #   2) Sum all the elements whose indexes are in set S and add the pooled value corresponding to the block.
    #   3) Divide the sum from step 2 by |S| + 1, where |S| is the cardinality of S.
    #   3) If an element whose index is not in S exceeds the latest mean from 3, repeat steps 1-4.
    #   4) Change all elements indexed in S to the latest mean from 3.
    #
    # ***COMPUTATIONAL APPROACH*** (equivalent to concept, but removes if-else logic)
    # For each block, do the following:
    #   1) Sort the elements of the blocks in descending order. Get inds. (such that sorted = unsorted[inds])
    #   2) Compute the cumulative sum, then add the pooled value corresponding to the block.
    #   3) Compute element-by-elemnt division of the sum from step 2 and [2,3,...,number of elements per block + 1]
    #   4) Find the maximum M from the values computed in step 3. Get the corresponding index maxInd.
    #   5) Change certain elements to M (unsortedNew = unsorted, unsortedNew[inds[0:maxInd + 1]] = M)

    concat_splits = lambda value,numOfSplits,splitDim,concatDim: tf.concat(tf.split(value,numOfSplits,splitDim),concatDim)

    pooledShape = x[1].get_shape().as_list()


    # vectorize blocks
    gatheredRows = concat_splits(value=x[0],numOfSplits=pooledShape[1],splitDim=1,concatDim=3)
    gatheredRows = concat_splits(value=x[1],numOfSplits=pooledShape[1],splitDim=1,concatDim=3)
    gatheredBlocks = concat_splits(value=gatheredRows,numOfSplits=pooledShape[2],splitDim=2,concatDim=3)
    gatheredBlocks2 = concat_splits(value=gatheredRows2,numOfSplits=pooledShape[2],splitDim=2,concatDim=3)
    blockShape = gatheredBlocks.get_shape().as_list()
    vectorizedBlocks = tf.reshape(gatheredBlocks,[-1,blockShape[1]*blockShape[2],1,blockShape[3]])

    # compute replacement values in least-squares solution and identify which elements to replace
    inds = tf.argsort(values=vectorizedBlocks,axis=1,direction='DESCENDING',stable=True)
    sortedblk = tf.sort(values=vectorizedBlocks,axis=1,direction='DESCENDING',stable=True)
    cumSum = tf.cumsum(x=sortedblk,axis=1) + tf.broadcast_to(gatheredBlocks2,sortedblk.get_shape().as_list())
    cumAvg = tf.multiply(cumSum,tf.const(np.reciprocal(np.arange(2.0,blockShape[1]*blockShape[2] + 2.0))))
    replacementVals = tf.max_reduce(cumAvg,axis=1,keepdims=True)
    argMax = tf.math.argmax(cumAvg,axis=1)
    selectInds = tf.less_equal(tf.range(0,blockShape[1]*blockShape[2]),argMax)

    # replace elements identified with computed replacement values
    splitSelectInds = tf.split(selectInds,blockShape[1]*blockShape[2],1)
    splitInds = tf.split(inds,blockShape[1]*blockShape[2],1)
    binaryTensor = tf.logical_and(splitSelectInds[0],tf.math.equal(splitInds[0],tf.range(0,blockShape[1]*blockShape[2])))
    for ii in range(1,len(splitSelectInds)):
      binaryTensor = tf.logical_or(binaryTensor,tf.logical_and(splitSelectInds[ii],tf.math.equal(splitInds[ii],tf.range(0,blockShape[1]*blockShape[2]))))
    #endfor
    vectorizedOutput = tf.where_v2(binaryTensor,replacementVals,vectorizedBlocks)

    # undo vectorization of blocks
    outputBlocks = tf.reshape(vectorizedOutput,block_shape)
    outputRows = concat_splits(value=outputBlocks,numOfSplits=pooledShape[2],splitDim=3,concatDim=2)
    outputRows2 = concat_splits(value=replacementVals,numOfSplits=pooledShape[2],splitDim=3,concatDim=2)
    output = concat_splits(value=outputRows,numOfSplits=pooledShape[1],splitDim=3,concatDim=1)
    output2 = concat_splits(value=outputRows2,numOfSplits=pooledShape[1],splitDim=3,concatDim=1)
    return (output,output2)
    
    
  def compute_output_shape(self, input_shape):
    return input_shape[0]

#endclass






class multilayerADMMsparseCodingTightFrame(Layer):

  def __init__(self,noc,kerSz,poolSz,rho,lambduh,noi,**kwargs):
    # noc = list of numbers of channels for each layer (excludes input)
    # kerSz = list of convolutional kernel sizes for each layer
    # poolSz = list of pooling sizes for each layer
    # rho = a positive constant scalar associated with ADMM algorithm
    # lambduh = the L1 penalty coefficient for last layer if scalar, or every layer if list
    # noi = number of iterations (for ADMM algorithm)
    self.noc = noc
    self.kerSz = kerSz
    self.poolSz = poolSz
    self.nol = len(kerSz)
    self.noi = noi
    self.rho = rho
    if isinstance(lambduh, list):
      self.lambduh = lambduh
    else:
      self.lambduh = [0]*len(poolSz) + lambduh
    #endif
    assert(len(noc) == len(kerSz))
    assert(len(kerSz) == len(poolSz) + 1)
    
    super(multilayerADMMsparseCodingTightFrame, self).__init__(*kwargs)

  def build(self, input_shape):
    paddingSz = []
    outputSz = []
    second_ind = lambda value,ind: [x[ii][ind] for ii in range(len(x))]
    floor_ceil = lambda value : [int(np.floor(value)),int(np.ceil(value))]

    # Compute padding size and size of output
    for jj in range(2):
      ps,os = get_padding_size(input_shape[1 + jj],second_ind(self.kerSz,jj),second_ind(self.poolSz,jj),self.nol)
      paddingSz.append(ps)
      outputSz.append(os)
    self.paddingSz = [[0,0],floor_ceil(paddingSz[0]/2.),floor_ceil(paddingSz[1]/2.),[0,0]]
    self.outputSz = outputSz

    # Add weights
    self.weights = []
    nof = inputShape[3]
    for ii in range(nol):
      self.weights.append(self.add_weight(shape=(self.kerSz[ii],nof,self.noc[ii]),
        initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.05),
        constraint=keras.constraints.UnitNorm(axis=2)))
      nof = self.noc[ii]
    self.nof = inputShape[3]
    super(multilayerADMMsparseCodingTightFrame, self).build(input_shape)

  def call(self, x):

    # Helpful naming conventions:
    # amg = alpha - gamma
    # zpg = z + gamma
    # dhypmu = D^H(y + mu)
    # dhdzpg is D^HD(z + gamma)
    # dammu = D(alpha) - mu
    # poolz = max_pool(z)
    # dhpoolzpmu = D^H(max_pool(z) + mu)
    
    paddedx = tf.pad(x,self.paddingSz,mode='CONSTANT')
    y = paddedx
    dh = lambda values, ind: tf.nn.conv2d(value=values,filter=self.weights[ind],padding="VALID")
    d = lambda values, ind: tf.nn.conv2d_transpose(value=values,filter=self.weights[ind],padding="VALID")
    max_pool = lambda values, ind: tf.keras.layers.MaxPool2d(pool_size=self.poolSz[ind],padding="valid")(values)
    alpha[0] = 1./2.*dh(y,0)
    z[0] = alpha[0]
    gamma[0] = alpha[0] - z[0]
    mu[0] = y - d(alpha[0],0)

    for ii in range(1,self.nol):
      poolz = max_pool(z[ii - 1],ii - 1)
      alpha[ii] = 1./2.*dh(poolz,ii)
      z[ii] = alpha[ii]
      gamma[ii] = alpha[ii] - z[ii]
      mu[ii] = poolz - d(alpha[ii],ii)

    binaryTensor = tf.pad(tf.constant(value=True,shape=x.get_shape()),self.paddingSz,mode='CONSTANT',constant_values=False)


    for iter in range(self.noi):
      temp1 = d(alpha[0],0) - mu[0]
      y = tf.where_v2(binaryTensor,1./(1 + self.rho)*(paddedx + self.rho*temp1),temp1)
      dhypmu = 1./2.*dh(y + mu[0],0)
      zpg = z[0] + gamma[0]
      dhdzpg = 1./2.*dh(d(zpg,0),0)
      alpha[0] = zpg + dhypmu - dhdzpg
      mu[0] = mu[0] + y - d(alpha[0],0)
      for ii in range(1,self.nol):
        amg = alpha[ii - 1] - gamma[ii - 1]
        dammu = d(alpha[ii],ii) - mu[ii]
        z[ii - 1],poolz = Unpool_LS(poolSz[ii - 1])(amg,dammu)
        # poolz = max_pool(z[ii - 1],ii - 1) # I modified Unpool_LS to output this so that it's not computed twice
        if self.lambduh[ii - 1] != 0:
          z[ii - 1] = shrinkage_layer(self.lambduh[ii - 1]/self.rho)(z[ii - 1])
          poolz = shrinkage_layer(self.lambduh[ii - 1]/self.rho)(poolz)
        #endif

        gamma[ii - 1] = gamma[ii - 1] + alpha[ii - 1] - z[ii - 1]        
        
        dhpoolzpmu = 1./2.*dh(poolz + mu[ii],ii)
        zpg = z[ii] + gamma[ii]
        dhdzpg = 1./2.*dh(d(zpg,ii),ii)
        alpha[ii] = dhpoolzpmu + zpg - dhdzpg

        mu[ii] = mu[ii] + poolz - d(alpha[ii],ii)
      #endfor
      z[self.nol - 1] = shrinkage_layer(self.lambduh[self.nol - 1]/self.rho)(mu[self.nol - 1])
      gamma[self.nol - 1] = z[self.nol - 1] - alpha[self.nol - 1]
    #endfor
    return z[self.nol - 1]


  def compute_output_shape(self, input_shape):
    return [input_shape[0],self.outputSz[0],self.outputSz[1],self.noc[-1]]
