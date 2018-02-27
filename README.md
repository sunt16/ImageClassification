# project
Classification problem 

matlab program here used to organize training data and test data

A simple cnn model has been used to classify airplanes, ferry and laptop images from imagenet

Place_holder layer

  x accept the input images, and y_ accept the labels which correspond to images

Convolution layer

  input: images set of size ?x128x128x3
  
  output: data set of size ?x64x64x32
  
  weights parameter size: 5x5x32, bias parameter size 32

Nonlinear transformation layer
  
  Relu nonlinear function

Pooling layer
  
  2x2 maximum pooling

Convolution layer
  
  input: data set of size ?x64x64x32
  
  output: data set of size ?x32x32x64
  
  weight parameter size: 5x5x64, bias parameter size 64

Nonlinear transformation layer

  Relu nonlinear function
  
Pooling layer

  2x2 maximum pooling

Fully connected layer

  input: data set of size ?x32x32x64 (?x65536)

  output: data set of size ?x1024
  
  weight parameter size: 65536x1024, bias parameter size 1024
  
Dropout layer
  
  input: data set of size ?x1024
  
  output: data set of size ?x1024
  
  keep_prob is 0.5 for training data, while 1 for test data
  
Fully connected layer
  
  input: data set of ?x1024
  
  output: data set of ?x3
  
  weight parameter size: 1024x3, bias parameter size: 3
  
  In this program, training data batch size is 10, which avoid the unstable gradient descent direction of loss function.
  
  reference: https://www.zhihu.com/question/32673260
  
  learning rate: 0.0001
  
  the reason of adamoptimizer has been used is that:
    
    1. Automatically adjust the learning rate
    
    2. Excellent performace for unstable loss function
    
    reference: https://www.jianshu.com/p/aebcaf8af76e
    
    Accuracy: 70% for the result of overfitting
    
