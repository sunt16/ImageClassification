# project
Classification problem 
matlab program here used to organize training data and test data

A simple cnn model has been used to classify airplanes, ferry and laptop images from imagenet

Place_holder layer

  x accept the input images, and y_ accept the labels which correspond to images

Convolution layer

  input: images set of size ?x128x128x3
  output: data set of size ?x64x64x3
  weights parameter size: 5x5x32, bias parameter size 32

Nonlinear transformation layer
  Relu nonlinear function
Pooling layer
  2x2 maximum pooling
Convolution layer
  input: 
Nonlinear transformation layer
Pooling layer
Fully connected layer
Dropout layer
Fully connected layer
