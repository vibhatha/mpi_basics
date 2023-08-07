## Simple Neural Network

### 1. Activation Function
We use the sigmoid function, $\( \sigma(z) \)$, as our activation function:
$\[ \sigma(z) = \frac{1}{1 + e^{-z}} \]$

The derivative of the sigmoid function, which is used during backpropagation, is:
$\[ \sigma'(z) = \sigma(z)(1 - \sigma(z)) \]$

### 2. Loss Function
Although I didn't explicitly specify a loss function in the code, the backpropagation process suggests that we're using the mean squared error (MSE) for a given set of predictions \( \hat{y} \) and true values \( y \):

$$
\[ L(\hat{y}, y) = \frac{1}{2N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2 \]
$$

Here, $\( N \)$ is the number of samples.

### 3. Forward Propagation
This describes the process of passing an input $\( X \)$ through the network to obtain the output $\( \hat{y} \)$. The intermediate steps are:
1. Hidden layer pre-activation:
$\[ z_1 = XW_1 + b_1 \]$
2. Hidden layer activation:
\[ a_1 = \sigma(z_1) \]
3. Output layer pre-activation:
\[ z_2 = a_1W_2 + b_2 \]
4. Output:
\[ \hat{y} = \sigma(z_2) \]

### 4. Backward Propagation
Backpropagation uses the chain rule from calculus to compute the gradient of the loss function with respect to each weight in the network. Here's how:

1. Output error:
\[ \delta_2 = \hat{y} - y \]
Given that we're using the MSE, this is the derivative of the loss with respect to \( z_2 \).

2. Hidden layer error:
\[ \delta_1 = \delta_2 \cdot W_2^T \times \sigma'(z_1) \]
Here, \( \times \) represents element-wise multiplication.

3. Gradients for the weights and biases:
\[ \frac{\partial L}{\partial W_2} = a_1^T \cdot \delta_2 \]
\[ \frac{\partial L}{\partial b_2} = \sum \delta_2 \]
\[ \frac{\partial L}{\partial W_1} = X^T \cdot \delta_1 \]
\[ \frac{\partial L}{\partial b_1} = \sum \delta_1 \]

### 5. Weight Update
After obtaining the gradients, the weights and biases are updated using gradient descent. For a learning rate \( \alpha \):

\[ W_2 = W_2 - \alpha \cdot \frac{\partial L}{\partial W_2} \]
\[ b_2 = b_2 - \alpha \cdot \frac{\partial L}{\partial b_2} \]
\[ W_1 = W_1 - \alpha \cdot \frac{\partial L}{\partial W_1} \]
\[ b_1 = b_1 - \alpha \cdot \frac{\partial L}{\partial b_1} \]

This iterative process of forward propagation, backpropagation, and weight update continues for a specified number of epochs or until the network's performance on the training data converges.

Remember, while the above is a thorough overview of the neural network presented, deep learning research has introduced many additional concepts, techniques, and optimizations that can be applied to more complex models and tasks.