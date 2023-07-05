# AI-ERA: Artificial Intelligence-Empowered Resource Allocation for LoRa-Enabled IoT Applications

In this paper, we propose a novel proactive approach---``artificial intelligence-empowered resource allocation’’ (AI-ERA)---to address the resource assignment issue in static and mobile IoT applications. The AI-ERA approach consists of two modes, namely offline and online modes. First, a deep neural network (DNN) model is trained with a dataset generated at ns-3 in the offline mode. Second, the proposed AI-ERA approach utilizes the pre-trained DNN model in the online mode to proactively assign an efficient SF for the end device before each uplink packet transmission.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See the published paper, Link: https://ieeexplore.ieee.org/abstract/document/10050828, for detailed instructions on how to deploy the project on a live LoRaWAN network in ns-3.

### Prerequisites

The code provided is designed to be compatible with PyTorch 1.7.1 and Python 3.8, ensuring optimal functionality and performance. PyTorch 1.7.1 is a powerful deep-learning framework that offers a wide range of tools and capabilities for building and training neural networks. Python 3.8, on the other hand, provides a robust and versatile programming environment. By leveraging the features of both PyTorch 1.7.1 and Python 3.8, the code can take advantage of the latest advancements in deep learning and harness the flexibility and ease of use offered by Python. Additionally, the code is also compatible with ns3.33, a popular network simulation framework, allowing seamless integration with network simulations and enabling comprehensive analysis and evaluation of network performance.

### Installing

A step-by-step series of examples that tell you how to get a development environment running!

Installing PyTorch
```
conda install pytorch==1.7.1 torchvision torchaudio cudatoolkit=10.1 -c pytorch
```
Installing ns-3 (Ubuntu 18.04 LTS recommended)
```
https://www.nsnam.org/wiki/Installation#Ubuntu.2FDebian.2FMint
```


## Running the Code

To run the code, follow these instructions.

### Main File

We consider the most straightforward ML algorithm called MLP. The MLP model comprised five fully connected layers for 6 classification classes (i.e., SF7 to SF12). The input layer receives the input data, a feature vector of 24 flattened features. The fully connected or hidden layers are responsible for learning and extracting features from the input data. Each neuron in a fully connected layer is connected to every neuron in the previous layer. Typically, an activation function, such as Rectified Linear Unit (ReLU), is applied after each fully connected layer to introduce non-linearity and enable the model to learn complex patterns in the data. The output layer is the final layer in the MLP model, which produces the predicted probabilities for each class. In this case, the output layer consists of a fully connected layer followed by a softmax activation function. The number of neurons in the output layer equals the number of classes, which is 6 in this scenario. The softmax activation function takes the output from the previous layer and converts it into a probability distribution over the classes. It ensures that the predicted probabilities sum up to 1 and represent the likelihood of the input belonging to each class. During training, the model adjusts its weights and biases using a suitable optimization algorithm (e.g., Adam) and a loss function (e.g., Cross-Entropy Loss) to minimize the discrepancy between the predicted probabilities and the true labels. Once trained, the model can be classified by predicting the class with the highest probability for a given input.

```
[your_env_path] python3 main_code.py
```

## Dataset

To understand the dataset generation and labeling, please refer to the https://ieeexplore.ieee.org/abstract/document/10050828. The dataset is in the ''data'' folder. Currently, the Python code inside the folder can generate a sequence of 6 groups (24 flattened features) required for MLP.

```
[your_env_path] python3 dataset_gen.py
```

## Deployment

Once the MLP model is trained, it could be utilized in the ns-3. In order to deploy it in the ns-3 LoRaWAN module, the inference model can be deployed either on the device or network server side based on the application requirements. In our paper, we have deployed it on the device side since we were more interested in the packet success ratio.
Once deployed and the required sequence is generated, the following code is used to run the inference mode.

```
system("[python_path] [inference_model_path]/inference.py");
```

## Authors

* **Arshad Farhad** - Ph.D.
