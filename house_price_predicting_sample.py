import tensorflow as tf
import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.animation as animation
#sess=tf.Session()
#generating some house sizes

num_house =160
np.random.seed(42) #seed() is used to create a numpy array
house_size=np.random.randint(low=1000,high=3500,size=num_house)
#we created the first numpy array of values

np.random.seed(42)
house_price=house_size*100.00+np.random.randint(low=20000,high=70000,size=num_house)

#using matplotlib model a representation of the above data

#plt.plot(house_size,house_price,"bx") #bx is used to represent them dots in blue
#plt.ylabel("Price")
#plt.xlabel("Size")
#plt.show()

#now normalize our data
def normalize(array):
    return (array-array.mean())/array.std()
        #(x-u)/std


#now devide the data set for training and testing as follows
num_train_samples=int(math.floor(num_house * 0.7))
#in above code the latest numpy verion deosn't allow floats as indexes,
# so we have to convert it to a int

#define training data

train_house_size=np.asanyarray(house_size[:num_train_samples])
train_house_price=np.asanyarray(house_price[:num_train_samples])

train_house_size_norm=normalize(train_house_size)
train_house_price_norm=normalize(train_house_price)


#define test data
test_house_size=np.array(house_size[num_train_samples:])
test_house_price=np.array(house_price[num_train_samples:])

test_house_size_norm=normalize(test_house_size)
test_house_price_norm=normalize(test_house_price)         

#type of tensors in TF
#Constant  -that value won't change when we execute the computations in the computation graph
#Variable  -that value changes when we execute the computations in the computation graph
#PlaceHolder -used to pass data into a graph at execution time like stdin

#now we have to create tensors to pass house_size and house_price to the graph
tf_house_size=tf.placeholder("float",name="house_size")
tf_house_prize=tf.placeholder("float",name="house_prize")

#now let's define variables holding the size_factor and price we set during the training
# we initialize them to some random values based on normal dist
# 
tf_size_factor=tf.Variable(np.random.randn(),name="size_factor") 
tf_price_offset=tf.Variable(np.random.randn(),name="price_offset")

#now here we implement the inferencing function using TF to predict the house price

tf_price_prediction=tf.add(tf.multiply(tf_size_factor,tf_house_size),tf_price_offset)
#since we are using linear regression to model the housing price above equation is like y=mx+c

#now define the Loss Function(how much error) -Mean Squared Error

tf_cost=tf.reduce_sum(tf.pow(tf_price_prediction-tf_house_prize,2))/(2*num_train_samples)

#optimize learning 

learning_rate =0.1

# define the gradient descent optimizer that will minimize the loss function making the tf_cost close to 0
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

#initializing the variables
init=tf.global_variables_initializer()
#launching the graph in the session
#with tf.Session() as sess :
 #   sess.run(init)

sess=tf.Session()
sess.run(init)
#set how often to display training progress and number of 
display_every = 2
num_training_iter=50

#keep iterating the training data
for iteration in range(num_training_iter):
        #fit all training data
    for(x,y) in zip(train_house_size_norm,train_house_price_norm):
        sess.run(optimizer,feed_dict={tf_house_size:x,tf_house_prize:y})

    if(iteration+1) % display_every ==0:
        c=sess.run(tf_cost,feed_dict={tf_house_size:train_house_size_norm,tf_house_prize:train_house_price_norm})
        print("iteration #:",'%04d'%(iteration+1),"cost =","{:.9f}".format(c),"size_factor=" ,sess.run(tf_size_factor),"price_offset=",sess.run(tf_price_offset))


print("Optimization Finished!!!!!")

train_house_size_mean=train_house_size.mean()
train_house_size_std =train_house_size.std()

train_house_price_mean=train_house_price.mean()
train_house_price_std=train_house_price.std()


plt.rcParams["figure.figsize"]=(10,8)
plt.figure()
plt.ylabel("Price")
plt.xlabel("Size(sq.ft)")
plt.plot(train_house_size,train_house_price,'go',label='Training Data')
plt.plot(train_house_size_norm*train_house_size_std+train_house_size_mean,
(sess.run(tf_size_factor)*train_house_size_norm+sess.run(tf_price_offset))*train_house_price_std+train_house_price_mean,label="Learned Regression")

plt.legend(loc='Upper Left')
plt.show()











