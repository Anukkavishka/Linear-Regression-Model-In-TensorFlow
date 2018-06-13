#this neural network will not have any hidden layers,
#and we are going to use it to recognize hand written characters
#we are using test data from MNIST dataset 

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#code above here let's you import the tf sites repository for datasets for tutorials
sess=tf.InteractiveSession()#in this Deep nueral network we use InteractiveSession()
#we use the TF helper function to pull down the data from the MNIST site
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

#we create a placeholder for 28*28 image data to be passed to our neural network
x=tf.placeholder(tf.float32,shape=[None,784]) 
y_=tf.placeholder(tf.float32,shape=[None,10])

#changing the MNIST input data from a list of values to a 28*28*1 grayscale valued cube
#which the Convolution NN can use:

x_image=tf.reshape(x,[-1,28,28,1],name="x_image")

#we are going to use RELU as our activation function and if the value of x<0 y=0;x>0 y=x

#define helper functions to created the weights and baises variables and convolution and pooling layers

#these must be initialized to a small positive number
#and with some noise you don't end up going to zero when comparing diffs

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
    
#convolution and the pooling we do Convolution,and then pooling to control overfitting

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x , ksize=[1,2,2,1],strides=[1,1,1,1],padding='SAME')


#define layers in the NN

#1st CNN
#32 features for each 5x5 patch of the image

W_conv1=weight_variable([5,5,1,32]) #1 para represents the input channels since it's gray scale 
                                    #we have only 1,if we were using RGB we will have 3
b_conv1=bias_variable([32])
#Do convolution on images,add bias and push through RELU activation
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
#take the results and run through max_pool
h_pool1=max_pool_2x2(h_conv1)

#2nd convolutional layer 
#process the 32 features from the first convolutional layer in 5x5 patch.
#Return 64 features weights and biases.

W_conv2=weight_variable([5,5,32,64]) 
b_conv2=bias_variable([64])

#Do convolution of the output of the 1st convolution latyer pool results
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
#take the results and run through max_pool
h_pool2=max_pool_2x2(h_conv2)

#defining fully connected layer

W_fulconnc1=weight_variable([7 * 7 * 64,1024]) 
b_fulconnc1=bias_variable([1024])

#connect output of pooling layer 2 as input for the fully connected layer
h_pool2_flat=tf.reshape(h_pool2, [-1 , 7 * 7 * 64] )
h_fulconnc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fulconnc1) + b_fulconnc1)

#since we are using backProp algorithm alos the model might overfit the testing data
#and the real world data will be predicted incorrectly,so we drop out few nuerons from the
#fully connected layer
#to solve this we use the below code
keep_prob=tf.placeholder(tf.float32) # get the drop out probability as a training input
h_fulconnc1_drop=tf.nn.dropout(h_fulconnc1,keep_prob)

#Readout layer
W_fulconnc2=weight_variable([1024,10]) 
b_fulconnc2=bias_variable([10])

#define model
y_conv=tf.matmul(h_fulconnc1_drop,W_fulconnc2) +b_fulconnc2



#now we have to create a function to measure the loss just like in previous models
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv,labels=y_))

#y_ is the result given value and the y_conv is the predicted value

#now let's optimize 
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#what is the correct prediction

correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))

accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#initialize all of the variables
sess.run(tf.global_variables_initializer())

#train the model
import time

#define num of steps and how often we display the progress

num_steps =3000
display_every=100

#starting the timer

start_time=time.time()
end_time=time.time()

for i in range(num_steps):
    batch=mnist.train.next_batch(50)
    train_step.run(feed_dict={x:batch[0] , y_:batch[1] , keep_prob:0.5})

    #periodic status display
    if i%display_every == 0 :
        train_accuracy=accuracy.eval(feed_dict={
             x:batch[0] , y_:batch[1] , keep_prob :1.0})
        end_time=time.time()
        print("step {0},elapsed time {1:.2f} seconds,training accuracy{2:.3f}% ".format(i,end_time-start_time,train_accuracy*100))


#display the summary

end_time=time.time()

print("Total training time for {0} batches : {1:.2f} seconds".format(i+1,end_time-start_time))

#test the accuracy on the test data

print("Test Accuracy {0:.3f}%".format(accuracy.eval(feed_dict={
x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0})*100.0))










