#this neural network will not have any hidden layers,
#and we are going to use it to recognize hand written characters
#we are using test data from MNIST dataset 

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#code above here let's you import the tf sites repository for datasets for tutorials
sess=tf.Session()
#we use the TF helper function to pull down the data from the MNIST site
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)

#we create a placeholder for 28*28 image data to be passed to our neural network
x=tf.placeholder(tf.float32,shape=[None,784]) 
#None means we dont know the dimension of the data set
#and 784 =28*28 valued vectors are going to be in there

#then we are going to create a y variable for our prediction and its a 10 element vector
y_=tf.placeholder(tf.float32,shape=[None,10])

#define weights and balances which will be changed when the model trains
w=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))

#defining our model to predict 
y=tf.nn.softmax(tf.matmul(x,w)+b)
#softmax funtion is an activation function for the neuron
#softmax function is a exponential function which means it highlights high values
#and surpresses the minim values


# now we have used prepared data -> defined our inference function
# lets create a loss measurement 
#  have used prepared data -> defined our inference function -> loss measurement

cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))

#with each training step we need to minimize the cross_entropy
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init=tf.global_variables_initializer()


sess.run(init)
#performing 1000 training steps
for i in range(1000) :
    batch_xs,batch_ys=mnist.train.next_batch(100)#get 100 data points from the date batch_xs=image
                                                 #batch_ys output number from [0-9]
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys}) #do the optimization with this data


#evaluate the model by comparing actual y and the predicted y_

correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1)) #outputs i fthe prediction is true/false

accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))#cast that output to 0/1

test_accuracy=sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels})

print("Test Accuracy : {0}%".format(test_accuracy*100))

sess.close()

#now this model is just used without any hidden layers so the perfomance is great but
#  not goog enough accuracy is emitted,so we are trying to implement the same model
#  with deep hidden layers.

#the predicted accuray of this model is just above 90%





