# Neural_network_from_scratch
Objective
The goal of this project is to implement a simple feedforward neural network in Python from scratch without using any deep learning libraries like TensorFlow or PyTorch. The implementation includes forward pass, backpropagation, and training using gradient descent.

Problem Definition
Dataset:
The dataset used is a binary classification problem based on the AND operation.

Inputs:

[0, 0], [0, 1], [1, 0], [1, 1]
Expected Outputs:

[0], [0], [0], [1]
Task:
The neural network is trained to predict the AND operation results for the given binary input combinations.

Neural Network Architecture
Input Layer:

2 neurons (representing two binary inputs).
Hidden Layer:

3 neurons.
Activation Function: Sigmoid.
Output Layer:

1 neuron (binary classification output).
Activation Function: Sigmoid.
Methodology
Forward Pass:
Multiply inputs with weights, add bias, and pass through the Sigmoid activation function (hidden layer).
Multiply hidden layer output with weights, add bias, and pass through Sigmoid (output layer).
Backpropagation:
Calculate error between predicted and actual outputs.
Update weights and biases using gradient descent.
Loss Function:
Mean Squared Error (MSE): Measures the squared difference between actual and predicted outputs.
Results
Sample Training Output:
yaml
Copy code
Epoch 0 - Loss: 0.3785
Epoch 1000 - Loss: 0.1035
Epoch 2000 - Loss: 0.0398
...
Epoch 9000 - Loss: 0.0022
Predictions After Training:
lua
Copy code
[[0.0011]
 [0.0441]
 [0.0393]
 [0.9356]]
Files in Repository
neural_network.py: The implementation of the neural network.
README.md: Documentation for the project.
Usage
Prerequisites:
Python 3.x
NumPy library
Running the Code:
Clone the repository:
bash
Copy code
git clone https://github.com/omvitole7/Neural_network_from_scratch.git
cd Neural_network_from_scratch
Run the script:
bash
Copy code
python neural_network.py
Output:
Training progress (loss at intervals).
Predictions for the AND problem after training.
License
This project is open-source and available under the MIT License.

Author
Om Vitole
GitHub: omvitole7
