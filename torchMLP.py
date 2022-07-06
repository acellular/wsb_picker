import numpy
from collections import OrderedDict
import torch
from torch.optim import SGD #stochastic gradient descent

#basic neural net
def simple_mlp_nn(features, hidden_dim, classes):
    mlpModel = torch.nn.Sequential(OrderedDict([
        ("hidden_layer", torch.nn.Linear(features, hidden_dim)),
        ("activation", torch.nn.ReLU()),
        ("output", torch.nn.Linear(hidden_dim, classes))]))
    return mlpModel

def next_batch(X, y, batch_size): #inputs (X), targets (y)
    # loop over the dataset
    for i in range(0, X.shape[0], batch_size): #loop data in batch-sized chunks
        # yield a tuple of the current batched data and labels
        yield (X[i:i + batch_size], y[i:i + batch_size])


# basic training and prediction function
# (remember must also put mlp into training or evaluation modes first)
def train_assess(X, y, mlp, dev="cpu", lr=1e-2, weight_decay=0):
    optimizer = SGD(mlp.parameters(), lr=lr, weight_decay=weight_decay)#WEIGHT DECAY IS REGULARIZATION
    loss_func = torch.nn.CrossEntropyLoss() #TODO--look up loss functions

    loss_score = 0 #initialize loss
    accuracy = 0 #initialize accuracy 
    samples = 0

    # flash data to the current dev, run it through our
    # model, and calculate loss
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
    loss_score += loss.item() * y.size(0)
    accuracy += (y_pred.max(1)[1] == y).sum().item()
    samples += y.size(0)
    return loss_score, accuracy, samples


# for more optimized training using smaller samples for each model iteration
def batch_train(X_train, X_test, y_train, y_test, batch_size=64, epochs=10,
    lr=1e-2, hidden_dim=2, classes=None, verbose=False, weight_decay=0):

    # can use CUDA?
    #A THING FOR DEBUGGING CUDA ERRORS
    #import os
    #os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] training using {}...".format(dev))

    # data shape
    num_rows, num_cols = X_train.shape
    print (num_cols)
    if classes==None: unique = numpy.unique(y_train)

    # convert to PyTorch tensors
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).float()
    y_test = torch.from_numpy(y_test).float()

    # initialize model and print shape
    mlp = simple_mlp_nn(num_cols, hidden_dim, len(unique)).to(dev)
    print(mlp)

    # create a template to summarize current training progress
    training_printout = "epoch: {} test loss: {:.3f} test accuracy: {:.3f}"

    # loop epochs
    for epoch in range(0, epochs):
        
        print(f"Epoch: {epoch + 1}...")

        # training mode
        loss_train = 0 #initialize loss
        accuracy_train = 0 #initialize accuracy 
        samples_train = 0
        mlp.train() #put into training mode so model parameters to be updated during backpropagation.
        # loop the current batch
        for (X_batch, y_batch) in next_batch(X_train, y_train, batch_size):
            loss, accuracy, samples = train_assess(X_batch, y_batch, mlp, dev=dev, lr=lr)
            loss_train += loss
            accuracy_train += accuracy
            samples_train += samples
        # display model progress on the current training batch
        if verbose: print('TRAIN: ', training_printout.format(epoch + 1, (loss_train / samples_train),
            (accuracy_train / samples_train)))
        
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
        # display model progress on the current test batch
        if verbose: print('TEST: ',training_printout.format(epoch + 1, (loss_test / samples_test),
            (accuracy_test / samples_test)))

    print(f'Training finished. Loss: {loss_test / samples_test} Accuracy: {accuracy_test / samples_test}')
    return mlp

if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    from sklearn.model_selection import train_test_split 
    X, y = make_blobs(n_samples=2000, n_features=100, centers=5, cluster_std=10)#, random_state=95)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7)#, random_state=95)

    mlp = batch_train(X_train, X_test, y_train, y_test,
        batch_size=64, epochs=10, lr=1e-2, hidden_dim=8, verbose=True)
            

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