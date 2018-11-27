# @LI YANZHE 
# Restricted Boltzmann Machine 
# Python Version: 3.7

import numpy as np

class RBM:
    
    def __init__(self, num_visible, num_hidden):
        self.num_hidden = num_hidden
        self.num_visible = num_visible

        # Initial weights uniformly between [-0.1*sqrt(6/(h+v)] and [0.1*sqrt(6/(h+v))];
        self.weights = np.random.uniform(
            low = -0.1 * np.sqrt(6. / (num_hidden + num_visible)),
            high = 0.1 * np.sqrt(6. / (num_hidden + num_visible)),
            size = (num_visible, num_hidden)
        )

        # Initial Bias as zero
        self.hidden_bias = np.zeros((1, num_hidden))
        self.visible_bias = np.zeros((1, num_visible))

        # Initial momentum velocity
        self.velocity_weights = np.zeros((num_visible, num_hidden))
        self.velocity_hidden = np.zeros((1, num_hidden))
        self.velocity_visible = np.zeros((1, num_visible))

    def _sigmoid(self, x):
        """
        Sigmoid function
        """
        return 1.0 / (1 + np.exp(-x))

    # Contrastive Divergence
    def cd(self, data, epoches = 1000, lr = 0.1):
        """
        Use contrastive divergence algorithm train the rbm.

        Parameters:
        ----------
        data : A numpy matrix where each row is a training example;
        epoches : the max numble of iterations
        lr : learning rate of each epoch
        """

        num_data = data.shape[0]
        batch_size = 50                     # use sgd with stochastic of batch size
        momentum_start = 0.9        # use momentum update
        momentum_end = 0.4

        for epoch in range(epoches):

            # generate a batch of data
            idxs = np.random.choice(num_data, size = batch_size)
            batch_data = data[idxs]
            """
            Positive CD phase
            """
            # compute probability P(h=1|x)
            positive_hidden_prob = self._sigmoid(batch_data @ self.weights + self.hidden_bias)

            # activate hidden units
            positive_hidden_units = positive_hidden_prob > np.random.rand(batch_size, self.num_hidden)
            
            # compute positive gradient by X.T * prob_hidden for weight
            positive_weights = batch_data.T @ positive_hidden_prob

            # compute positive gradient for bias
            positive_hidden = np.sum(positive_hidden_prob, axis=0)
            positive_visible = np.sum(batch_data, axis=0)

            """
            Negtive CD phase
            """
            # Gibbs Sampling
            negtive_visible_prob = self._sigmoid(positive_hidden_units @ self.weights.T + self.visible_bias)

            # Compute probability P(h =1 | x_hat)
            negtive_hidden_prob = self._sigmoid(negtive_visible_prob @ self.weights + self.hidden_bias)

            # compute negtive gradient by X_hat.T * prob_hidden
            negtive_weights = negtive_visible_prob.T @ negtive_hidden_prob

            # compute negtive gradient for bias
            negtive_hidden = np.sum(negtive_hidden_prob, axis=0)
            negtive_visible = np.sum(negtive_visible_prob, axis=0)
            
            """
            Update weights and bias
            """
            if epoch < epoches / 2:
                mu = momentum_start
            else:
                mu = momentum_end
            
            #Update Velocity
            self.velocity_weights = mu * self.velocity_weights + lr * (positive_weights - negtive_weights) / batch_size
            self.velocity_hidden = mu * self.velocity_hidden + lr * (positive_hidden - negtive_hidden) / batch_size
            self.velocity_visible = mu * self.velocity_visible + lr * (positive_visible - negtive_visible) / batch_size
            
            # Update weights and bias
            self.weights += self.velocity_weights
            self.hidden_bias += self.velocity_hidden
            self.visible_bias += self.velocity_visible

            # compute error
            error = np.sum((batch_data - negtive_visible_prob) ** 2)
            
            # Print Debug message
            if epoch % 100 == 0:
                print(f'Epoch {epoch}: Error is {error}')


    def reconstruct(self, data, epoches=1):
        """
        Reconstruct the data use this rbm.
        Paramter:
        ----------
        data : A numpy matrix where each row is a Example.
        epoches : the number of reconstruct iteration
        Return: 
        ----------
        visible_probs : the reconstructed data, has the same number of examples with data
        """

        num_data = data.shape[0]
        visible_units = data

        for epoch in range(epoches):
            # compute hidden units
            hidden_prop = self._sigmoid(visible_units @ self.weights + self.hidden_bias)
            hidden_units = hidden_prop > np.random.rand(num_data, self.num_hidden)

            # reconstruction use hidden units
            visible_prob = self._sigmoid(hidden_units @ self.weights.T + self.visible_bias)
            visible_units = visible_prob > np.random.rand(num_data, self.num_visible)

        return visible_prob, visible_units