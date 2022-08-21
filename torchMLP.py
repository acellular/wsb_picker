import numpy
from collections import OrderedDict
import torch
from torch.optim import SGD #stochastic gradient descent

# basic neural net
def simple_mlp_nn(features, nodes, classes):
    mlpModel = torch.nn.Sequential(OrderedDict([
        ("input_layer", torch.nn.Linear(features, nodes)),
        ("activation", torch.nn.ReLU()),
        ("output", torch.nn.Linear(nodes, classes))]))
    return mlpModel

# mlp with easily definable layer sizes (like sklearn's MLP)
def mlp_nn(features, classes, hidden_layer_sizes=(200,100)):
    layers = OrderedDict()
    layers['input_layer'] = torch.nn.Linear(features, hidden_layer_sizes[0])
    layers['activation_input'] = torch.nn.ReLU()
    for i in range(1,len(hidden_layer_sizes)):
        layers[f'hidden_layer_{i}'] = torch.nn.Linear(hidden_layer_sizes[i-1], hidden_layer_sizes[i])
        layers[f'activation_{i}'] = torch.nn.ReLU()

    layers['output'] = torch.nn.Linear(hidden_layer_sizes[len(hidden_layer_sizes)-1], classes)
    mlpModel = torch.nn.Sequential(layers)
    return mlpModel

# TODO--a model for 2D+ data that starts with Flatten()?

def next_batch(X, y, batch_size): #inputs (X), targets (y)
    # loop over the dataset
    for i in range(0, X.shape[0], batch_size): #loop data in batch-sized chunks
        # yield a tuple of the current batched data and labels
        yield (X[i:i + batch_size], y[i:i + batch_size])


# basic training and prediction function
# (remember must also put mlp into training or evaluation modes first)
def train_assess(X, y, mlp, dev="cpu", lr=1e-2, weight_decay=0):
    optimizer = SGD(mlp.parameters(), lr=lr, weight_decay=weight_decay)#WEIGHT DECAY IS REGULARIZATION
    #loss_func = torch.nn.L1Loss() #TODO--look up more loss functions
    loss_func = torch.nn.CrossEntropyLoss() 

    # send data to device, run model, calculate loss
    X, y = (X.to(dev), y.to(dev))
    y_pred = mlp(X)#THE ACTUAL NEURAL NET!
    loss = loss_func(y_pred, y.long())
    
    # important to be done in this order so gradients aren't carried
    # over from previous steps.
    if mlp.training:
        optimizer.zero_grad() #zero the gradient
        loss.backward() #backpropagation
        optimizer.step() #update model parameters (i.e. weights)
    
    # update training loss, accuracy, and # samples visited
    loss_score = loss.item() * y.size(0)
    accuracy = (y_pred.max(1)[1] == y).sum().item()
    samples = y.size(0)
    return loss_score, accuracy, samples


# for more optimized training using smaller samples for each model iteration
def batch_train(X_train, X_test, y_train, y_test, batch_size=64, epochs=10,
    lr=1e-2, hidden_layer_sizes=[2], classes=None, verbose=False, weight_decay=0):

    # can use CUDA?
    #A THING FOR DEBUGGING CUDA ERRORS
    #import os
    #os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev}")

    # data shape
    num_rows, num_cols = X_train.shape
    if classes==None: unique = numpy.unique(y_train)

    # convert to PyTorch tensors
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).float()
    y_test = torch.from_numpy(y_test).float()

    # initialize model and print shape
    mlp = mlp_nn(num_cols, len(unique), hidden_layer_sizes=hidden_layer_sizes).to(dev)
    print(mlp)

    # loop epochs
    for epoch in range(0, epochs):
        
        if epoch % 10 == 0: print(f"Epoch {epoch}...")

        # training mode
        loss_train = 0
        accuracy_train = 0
        samples_train = 0
        mlp.train() #put into training mode so model parameters updated during backpropagation.
        # loop the current batch
        for (X_batch, y_batch) in next_batch(X_train, y_train, batch_size):
            loss, accuracy, samples = train_assess(X_batch, y_batch, mlp, dev=dev, lr=lr)
            loss_train += loss
            accuracy_train += accuracy
            samples_train += samples
        
        # evaluation mode
        loss_test = 0
        accuracy_test = 0
        samples_test = 0
        mlp.eval()#switch in to eval so no gradient change
        with torch.no_grad(): #gradient computation turned off TODO--actually needed if already in eval?
            for (X_batch, y_batch) in next_batch(X_test, y_test, batch_size): #TEST DATA NOW!
                loss, accuracy, samples = train_assess(X_test, y_test, mlp, dev=dev, lr=lr)
                loss_test += loss
                accuracy_test += accuracy
                samples_test += samples
        # display model progress on current test batch
        if verbose: print(f'Epoch {epoch}: Train loss: {loss_train / samples_train:.3f} accuracy: {accuracy_train / samples_train:.3f},',
            f'Test loss: {loss_test / samples_test:.3f} accuracy: {accuracy_test / samples_test:.3f}')
    print(f'Training finished. Loss: {loss_test / samples_test} Accuracy: {accuracy_test / samples_test}')
    return mlp


# testing
if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    from sklearn.model_selection import train_test_split 
    X, y = make_blobs(n_samples=2000, n_features=100, centers=5, cluster_std=10)#, random_state=95)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7)#, random_state=95)

    mlp = batch_train(X_train, X_test, y_train, y_test,
        batch_size=64, epochs=10, lr=1e-2, hidden_layer_sizes=(2000,3000,100), verbose=True)
            

"""
from stack exchange on zeroing gradients:
In PyTorch, for every mini-batch during the training phase,
we typically want to explicitly set the gradients to zero before starting to do backpropragation
(i.e., updating the Weights and biases) because PyTorch accumulates the gradients on subsequent backward passes.
This accumulating behaviour is convenient while training RNNs or when we want to compute the gradient of the
loss summed over multiple mini-batches. So, the default action has been set to accumulate (i.e. sum)
the gradients on every loss.backward() call.

Because of this, when you start your training loop,
ideally you should zero out the gradients so that you do the parameter update correctly.
Otherwise, the gradient would be a combination of the old gradient,
which you have already used to update your model parameters, and the newly-computed gradient.
It would therefore point in some other direction than the intended direction towards
the minimum (or maximum, in case of maximization objectives).
"""