import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow import math
import numpy as np
import tensorflow.keras.constraints

def floor_ceil(value):
  # returns list of int: [floor(value),ceil(value)]
  return [int(np.floor(value)),int(np.ceil(value))]

def evenly_cropped(x,xRecon):
  #crops xRecon to the shape of x, trimming evenly from top and bottom, left and right. (In case of odd crop, top and left cropped less than bottom and right).
  inputShape = x.get_shape().as_list()
  reconShape = xRecon.get_shape().as_list()
  cropShape = (floor_ceil(np.float32(reconShape[1] - inputShape[1])/2.0),floor_ceil(np.float32(reconShape[2] - inputShape[2])/2.0))
  return tf.keras.layers.Cropping2D(cropping=cropShape)(xRecon)

def cropped_l2_loss(x,xRecon):
  # computes l2 loss between x and its reconstruction after cropping reconstruction to the size of x.
  croppedRecon = evenly_cropped(x,xRecon)
  return tensorflow.nn.l2_loss(croppedRecon)

def get_effective_filter_size(kerSz,poolSz,nol):
  # Can only handle 1-dimension. Please loop and call multiple times if multiple dimensions are necessary.
  filterSz = 1
  for ii in range(nol-1,0,-1):
    filterSz = (filterSz + kerSz[ii] - 1)*poolSz[ii - 1]
  return filterSz + kerSz[0] - 1

def get_padding_size(inputSz,kerSz,poolSz,nol):
  # computes how much to pad the input and the size of the last layer. Can only handle one dimension.
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

def concat_splits(value,numOfSplits,splitDim,concatDim):
  return tf.concat(tf.split(value,numOfSplits,splitDim),concatDim)

def get_gathered_blocks(x,numOfBlocks):
  gatheredRows = concat_splits(value=x,numOfSplits=numOfBlocks[0],splitDim=1,concatDim=3)
  return concat_splits(value=gatheredRows,numOfSplits=numOfBlocks[1],splitDim=2,concatDim=3)

def ungather_blocks(x,numOfBlocks):
    outputRows = concat_splits(value=x,numOfSplits=numOfBlocks[1],splitDim=3,concatDim=2)
    return concat_splits(value=outputRows,numOfSplits=numOfBlocks[0],splitDim=3,concatDim=1)




class shrinkage_layer(Layer):

  def __init__(self,lambduh,**kwargs):
    #lambduh must be scalar
    self.lambduh = lambduh
    super(shrinkage_layer, self).__init__(**kwargs)

  def build(self, input_shape):
    super(shrinkage_layer, self).build(input_shape)

  def call(self, x):
    y = x - self.lambduh*tf.math.sign(x)
    batchSize = tf.shape(x)[0]
    inputShape = x.get_shape().as_list()[1:]
    return tf.where(tf.math.logical_or(x <= -self.lambduh, x >= self.lambduh),y,tf.fill(dims=[batchSize,*inputShape],value=0.))

  def compute_output_shape(self, input_shape):
    return input_shape

class adaptive_shrinkage_layer(Layer):
  def __init(self,lambduh,**kwargs):
    self.lambduh = lambduh
    super(adaptive_shrinkage_layer, self).__init__(**kwargs)

  def build(self, input_shape):
    super(shrinkage_layer, self).build(input_shape)

  def call(self, x):
    lambduh = self.lambduh*x[1]
    y = x[0] - lambduh*tf.math.sign(x[0])
    batchSize = tf.shape(x[0])[0]
    inputShape = x[0].get_shape().as_list()[1:]
    return tf.where(tf.math.logical_or(x <= -lambduh, x >= lambduh),y,tf.fill(dims=[batchSize,*inputShape],value=0.))

  def compute_output_shape(self, input_shape):
    return input_shape[0]



class Unpool_LS(Layer):

  def __init__(self, poolSz, **kwargs):
    # poolSz is purely used for assertions made in the build method.
    self.poolSz = poolSz
    super(Unpool_LS, self).__init__(**kwargs)

  def build(self, input_shape):
    inputShape = input_shape[0]
    pooledShape = input_shape[1]
    assert(inputShape[1] == self.poolSz[0]*pooledShape[1])
    assert(inputShape[2] == self.poolSz[1]*pooledShape[2])
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

    pooledShape = x[1].get_shape().as_list()


    # vectorize blocks
    gatheredBlocks = get_gathered_blocks(x[0],(pooledShape[1],pooledShape[2]))
    blockShape = gatheredBlocks.get_shape().as_list()
    blockSize = blockShape[1]*blockShape[2]
    vectorizedBlocks = tf.reshape(gatheredBlocks,[-1,blockSize,1,blockShape[3]])
    vectorizedBlocks2 = get_gathered_blocks(x[1],(pooledShape[1],pooledShape[2]))
    
#    gatheredRows = concat_splits(value=x[0],numOfSplits=pooledShape[1],splitDim=1,concatDim=3)
#    gatheredRows2 = concat_splits(value=x[1],numOfSplits=pooledShape[1],splitDim=1,concatDim=3)
#    gatheredBlocks = concat_splits(value=gatheredRows,numOfSplits=pooledShape[2],splitDim=2,concatDim=3)
#    gatheredBlocks2 = concat_splits(value=gatheredRows2,numOfSplits=pooledShape[2],splitDim=2,concatDim=3)
#    blockShape = gatheredBlocks.get_shape().as_list()
#    blockSize = blockShape[1]*blockShape[2]
#    vectorizedBlocks = tf.reshape(gatheredBlocks,[-1,blockSize,1,blockShape[3]])

    # compute replacement values in least-squares solution and identify which elements to replace
    inds = tf.argsort(values=vectorizedBlocks,axis=1,direction='DESCENDING')
    sortedblk = tf.sort(values=vectorizedBlocks,axis=1,direction='DESCENDING')
    cumSum = tf.cumsum(x=sortedblk,axis=1) + vectorizedBlocks2
    cumAvg = tf.multiply(cumSum,tf.constant(np.reciprocal(np.arange(2.0,blockSize + 2.0)),dtype=tf.float32,shape=(1,blockSize,1,1)))
    replacementVals = tf.math.reduce_max(cumAvg,axis=1,keepdims=True)
    argMax = tf.reshape(tf.math.argmax(cumAvg,axis=1),[-1,1,1,blockShape[3]])
    blockIndsTensorConst = tf.reshape(tf.range(0,blockSize),[1,blockSize,1,1])
    selectInds = tf.less_equal(blockIndsTensorConst,tf.dtypes.cast(argMax,tf.int32))

    # replace elements identified with computed replacement values
    splitSelectInds = tf.split(selectInds,blockSize,1)
    splitInds = tf.split(inds,blockSize,1)
    binaryTensor = tf.logical_and(splitSelectInds[0],tf.math.equal(splitInds[0],blockIndsTensorConst))
    for ii in range(1,len(splitSelectInds)):
      binaryTensor = tf.logical_or(binaryTensor,tf.logical_and(splitSelectInds[ii],tf.math.equal(splitInds[ii],blockIndsTensorConst)))
    #endfor
    desiredShape = vectorizedBlocks.get_shape().as_list()[1:4]
    batchSize = tf.shape(vectorizedBlocks)[0]
    vectorizedOutput = tf.where(binaryTensor,tf.broadcast_to(replacementVals,[batchSize,*desiredShape]),vectorizedBlocks)
    vectorizedOutput3 = tf.where(binaryTensor,tf.broadcast_to(tf.dtypes.cast((argMax + 1)/(argMax + 2),tf.float32),[batchSize,*desiredShape]),tf.fill(dims=[batchSize,*desiredShape],value=np.float32(1.)))

    # undo vectorization of blocks
    outputBlocks = tf.reshape(vectorizedOutput,[batchSize,*blockShape[1:4]])
    outputBlocks3 = tf.reshape(vectorizedOutput3,[batchSize,*blockShape[1:4]])
    output = ungather_blocks(outputBlocks,(pooledShape[1],pooledShape[2]))
    output3 = ungather_blocks(outputBlocks3,(pooledShape[1],pooledShape[2]))
    output2 = ungather_blocks(replacementVals,(pooledShape[1],pooledShape[2]))
    #outputRows = concat_splits(value=outputBlocks,numOfSplits=pooledShape[2],splitDim=3,concatDim=2)
    #outputRows2 = concat_splits(value=replacementVals,numOfSplits=pooledShape[2],splitDim=3,concatDim=2)
    #outputRows3 = concat_splits(value=outputBlocks3,numOfSplits=pooledShape[2],splitDim=3,concatDim=2)
    #output = concat_splits(value=outputRows,numOfSplits=pooledShape[1],splitDim=3,concatDim=1)
    #output2 = concat_splits(value=outputRows2,numOfSplits=pooledShape[1],splitDim=3,concatDim=1)
    #output3 = concat_splits(value=outputRows3,numOfSplits=pooledShape[1],splitDim=3,concatDim=1)
    return (output,output2,output3)
    
    
  def compute_output_shape(self, input_shape):
    return (input_shape[0],input_shape[1],input_shape[0])

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
      self.lambduh = [0]*len(poolSz) + [lambduh]
    #endif
    assert(len(noc) == len(kerSz))
    assert(len(kerSz) == len(poolSz) + 1)
    
    super(multilayerADMMsparseCodingTightFrame, self).__init__(*kwargs)

  def build(self, input_shape):
    inputShape = input_shape.as_list()
    paddingSz = []
    outputSz = []
    second_ind = lambda value,ind: [value[ii][ind] for ii in range(len(value))]
    #floor_ceil = lambda value : [int(np.floor(value)),int(np.ceil(value))]

    # Compute padding size and size of output
    for jj in range(2):
      ps,os = get_padding_size(inputShape[1 + jj],second_ind(self.kerSz,jj),second_ind(self.poolSz,jj),self.nol)
      paddingSz.append(ps)
      outputSz.append(os)
    self.paddingSz = [[0,0],floor_ceil(paddingSz[0]/2.),floor_ceil(paddingSz[1]/2.),[0,0]]
    self.outputSz = outputSz

    # Add weights
    #self.weights = []
    nof = inputShape[3]
    for ii in range(self.nol):
      weightNames = "ADMM_convolutional_weights" + str(ii)
      weightShape = tf.constant(value=[self.kerSz[ii][0],self.kerSz[ii][1],nof,self.noc[ii]])
      self.weights.append(self.add_weight(name=weightNames,
        shape=[self.kerSz[ii][0],self.kerSz[ii][1],nof,self.noc[ii]],#weightShape,
        initializer=tf.keras.initializers.RandomNormal(mean=0.0,stddev=np.sqrt(np.sqrt(1./(nof*self.kerSz[ii][0]**2*self.kerSz[ii][1]**2*self.noc[ii]))))
        #initializer= tf.keras.initializers.RandomNormal(mean=0.0,stddev=np.sqrt(2/(self.kerSz[ii][0]*self.kerSz[ii][1]*self.noc[ii])))
        #initializer=tf.keras.initializers.Orthogonal(gain=1.0),
        #constraint=tf.keras.constraints.UnitNorm(axis=3)
        #constraint=tf.keras.constraints.MaxNorm(max_value=1.,axis=3)
        ))
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
    batchSize = tf.shape(x)[0]
    inputDimensions = x.get_shape()[1:4]
    binaryTensorShape = (batchSize,inputDimensions[0],inputDimensions[1],inputDimensions[2])
    binaryTensor = tf.pad(tf.fill(dims=binaryTensorShape,value=True),self.paddingSz,mode='CONSTANT',constant_values=False)
    #paddedxShape = tf.shape(paddedx) 
    #outshape = []
    # outshape.append(tf.pack([batch_size,paddedxShape[1],paddedxShape[2],paddedxShape[3]]))
    #for ii in range(self.nol)
    #  pass#outshape.append(
    outshape = lambda x, w: tf.stack([batchSize, x.get_shape().as_list()[1] + w.get_shape().as_list()[0] - 1, x.get_shape().as_list()[2] + w.get_shape().as_list()[1] - 1,w.get_shape().as_list()[2]])
    dh = lambda value, ind: tf.nn.conv2d(input=value,filter=self.weights[ind],strides=[1,1,1,1],padding="VALID")
    d = lambda input, ind: tf.nn.conv2d_transpose(value=input,filter=self.weights[ind],output_shape=outshape(input,self.weights[ind]),strides=[1,1,1,1],padding="VALID")
    max_pool = lambda values, ind: tf.keras.layers.MaxPool2D(pool_size=self.poolSz[ind],padding="valid")(values)
    alpha = []
    z = []
    gamma = []
    mu = []
    alpha.append(1/2.*dh(y,0))
    z.append(alpha[0])
    gamma.append(z[0] - alpha[0])
    mu.append(y - d(alpha[0],0))

    for ii in range(1,self.nol):
      poolz = max_pool(z[ii - 1],ii - 1)
      alpha.append(1/2.*dh(poolz,ii))
      z.append(alpha[ii])
      gamma.append(alpha[ii] - z[ii])
      mu.append(poolz - d(alpha[ii],ii))

    #binaryTensor = tf.pad(tf.constant(value=True,shape=tf.stack([1,x.get_shape()[1:4]])),self.paddingSz,mode='CONSTANT',constant_values=False)


    for iter in range(self.noi):
      temp1 = d(alpha[0],0) - mu[0]
      y = tf.where(binaryTensor,1./(1 + self.rho)*(paddedx + self.rho*temp1),temp1)
      dhypmu = 1./2.*dh(y + mu[0],0)
      zpg = z[0] + gamma[0]
      dhdzpg = 1./2.*dh(d(zpg,0),0)
      alpha[0] = zpg + dhypmu - dhdzpg
      mu[0] = mu[0] + y - d(alpha[0],0)
      for ii in range(1,self.nol):
        amg = alpha[ii - 1] - gamma[ii - 1]
        dammu = d(alpha[ii],ii) - mu[ii]
        z[ii - 1],poolz,shrinkageFactor = Unpool_LS(self.poolSz[ii - 1])([amg,dammu])
        
        if self.lambduh[ii - 1] != 0:
          z[ii - 1] = adaptive_shrinkage_layer(self.lambduh[ii - 1]/self.rho)([z[ii - 1],shrinkageFactor])
          poolz = max_pool(z[ii - 1],ii - 1)
        #endif

        gamma[ii - 1] = gamma[ii - 1] + z[ii - 1] - alpha[ii - 1]        
        
        dhpoolzpmu = 1./2.*dh(poolz + mu[ii],ii)
        zpg = z[ii] + gamma[ii]
        dhdzpg = 1./2.*dh(d(zpg,ii),ii)
        alpha[ii] = dhpoolzpmu + zpg - dhdzpg

        mu[ii] = mu[ii] + poolz - d(alpha[ii],ii)
      #endfor
      z[self.nol - 1] = shrinkage_layer(self.lambduh[self.nol - 1]/self.rho)(alpha[self.nol - 1] - gamma[self.nol - 1])
      gamma[self.nol - 1] = gamma[self.nol - 1] + z[self.nol - 1]  - alpha[self.nol - 1]
    #endfor
    return z[self.nol - 1],y


  def compute_output_shape(self, input_shape):
    return ([input_shape[0],self.outputSz[0],self.outputSz[1],self.noc[-1]],input_shape)


#def multilayerADMMVisualizationNetwork(nol,model,inputShape)
#  inputLayer = tensorflow.keras.Input(inputShape)
#  return visualizationModel
