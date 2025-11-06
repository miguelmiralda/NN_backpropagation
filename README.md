# NN_backpropagation

Miguel Miralda - i6319047
Jesse Lemeer - i6328583

NNJesse.ipynb:

In this jupyter notebook, numpy is used for input, weights, biases initialization and some math operations. 
Matplotlib is used for plotting.

To run the notebook, execute all cells sequentially.

Started with initializing input X and output Y, which are both just an 8 by 8 identity matrix (8 learning examples).
There is no split between train and test set, as we are trying to map the same input as output in this assignment. 
This means no generalizing.

There 8 input nodes, 3 hidden nodes, 8 output nodes. To initialize weights, we need a matrix w1 which has size: number of input nodes by
number of hidden node (8x3). And a matrix w2 which has size: number of hidden nodes by number of output nodes (3x8). 
The biases for the hidden layer is a vector of size 3, one for each hiddennode. The biases for the output layer is a vector of size 8, one for each output node. 
These values (biases and weights) are randomized with a normal distribution (like in the Backprop help pdf) between -0.1 and 0.1 to break symmetry and to let each node learn different features (this is explained more in the report).

Some functions are defined next. Sigmoid is our activation function, as we want the 8x8 identity matrix as output and sigmoid returns values between 0 and 1. 
The other common option for backpropagation, tanh (outputs between -1 and 1), might output negative values. 
There a function that calculates the weighted input, as I thought "z" was need in order to calculate delta. Afterwards I noticed the activation a does the job in calculating delta, you just don't have to put it in the sigmoid again. 
The loss function used is the half mean squared error (same as in Back prop help pdf). This is the cost function used as well for the gradient descent. Its derivate is simpler with the half in front that cancels out the 2. Another loss function could have been used, as said in the lab it does not matter. 

One iteration is explained, but many iterations are needed for convergence, which is when the loss is below 0.01 in this case. Some alpha is picked as learning rate.

In order to learn, we start with a feedforward pass that computes the activations for each layer (hidden and output). Then the backpropagation starts. For each unit in the output layer, delta is calculated by calling the delta_output function, which can be seen as the “error” of that unit. Then the delta's in the hidden layer are calculated, based on the delta in the output layer and calling the delta_hidden function. Then the gradients of both weight and bias for both the hidden and output layer is calculated based on the delta (and activation for weights). For the weights, the dot product of the activation in previous layer and the delta of the next layer is used. For the biases, the deltas across the learning examples are summed, since each bias corresponds to one node of that layer.
These gradients are used to update the respective weights and biases by substracting it multiplied by the learning rate from the current weights and biases. With these updated weights, a new iteration can start. 

We append per iterations, the loss and iteration number, for the plots. Once converged, the weights and activations are printed for interpretation and stop the process. 

The last two cells are for comparison of different learning rates, but use the same process. It keeps track of the amount of iterations for each learning rate in the list.

