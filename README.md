# project
Classification problem 

matlab program here used to organize training data and test data

Traning and test data: http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz

Reference: https://cn.mathworks.com/examples/matlab-computer-vision/mw/vision-ex77068225-image-category-classification-using-deep-learning?s_tid=srchtitle

A simple cnn model has been used to classify airplanes, ferry and laptop images from imagenet

![image](https://github.com/sunt16/ImageClassification/blob/master/picture/pic.png)

Place_holder layer

![image](https://github.com/sunt16/ImageClassification/blob/master/picture/pic2.png)

  x accept the input images, and y_ accept the labels which correspond to images

Convolution layer

![image](https://github.com/sunt16/ImageClassification/blob/master/picture/pic3.png)
 
  input: images set of size ?x128x128x3
  
  output: data set of size ?x64x64x32
  
  weights parameter size: 5x5x3x32, bias parameter size 32

Nonlinear transformation layer

  Relu nonlinear function

Pooling layer
   
  2x2 maximum pooling

Convolution layer

![image](https://github.com/sunt16/ImageClassification/blob/master/picture/pic4.png)

  input: data set of size ?x64x64x32
  
  output: data set of size ?x32x32x64
  
  weight parameter size: 5x5x32x64, bias parameter size 64

Nonlinear transformation layer

  Relu nonlinear function
  
Pooling layer

  2x2 maximum pooling

Fully connected layer

![image](https://github.com/sunt16/ImageClassification/blob/master/picture/pic5.png)

  input: data set of size ?x32x32x64 (?x65536)

  output: data set of size ?x1024
  
  weight parameter size: 65536x1024, bias parameter size 1024
  
Dropout layer
  
![image](https://github.com/sunt16/ImageClassification/blob/master/picture/pic6.png)
  
  input: data set of size ?x1024
  
  output: data set of size ?x1024
  
  keep_prob is 0.5 for training data, while 1 for test data
  
Fully connected layer

![image](https://github.com/sunt16/ImageClassification/blob/master/picture/pic7.png)

  input: data set of ?x1024
  
  output: data set of ?x3
  
  weight parameter size: 1024x3, bias parameter size: 3
  
  In this program, training data batch size is 10, which avoid the unstable gradient descent direction of loss function.
  
  reference: https://www.zhihu.com/question/32673260
  
  learning rate: 0.0001
  
  training number: 5000 times
  
  the reason of adamoptimizer has been used is that:
    
    1. Automatically adjust the learning rate
    
    2. Excellent performace for unstable loss function
    
  reference: https://www.jianshu.com/p/aebcaf8af76e
    
  Accuracy: almost euqual to 100%
  
  Curve of Loss function:
  
  ![image](https://github.com/sunt16/ImageClassification/blob/master/picture/pic1.png)
    
Save trained model
  
  During the training process, key variables have been saved, like the value of weight, bias and computation graph structure.
  
  These files can be found in https://github.com/sunt16/ImageClassification/tree/master/SavedModel
  
Predict the category of new pictures using the trained model
  
  Code in https://github.com/sunt16/ImageClassification/blob/master/cnn_prediction.py
  
  Part of the forcast result:
  
  Theis picture shows a ferry on the sea, the trained model gives the tag ferry, correct!
  
  ![image](https://github.com/sunt16/ImageClassification/blob/master/picture/pic8.png)
  
  This picture shows a laptop, the trained model gives the tag laptop, correct!
  
  ![image](https://github.com/sunt16/ImageClassification/blob/master/picture/pic9.png)
  
  This picture shows an airplane, the trained model gives the tag airplane, correct! 
  
  ![image](https://github.com/sunt16/ImageClassification/blob/master/picture/pic11.png)
