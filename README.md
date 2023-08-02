# AI-ERA: Artificial Intelligence-Empowered Resource Allocation for LoRa-Enabled IoT Applications

This paper proposes a novel proactive approach---``artificial intelligence-empowered resource allocation’’ (AI-ERA)---to address the resource assignment issue in static and mobile IoT applications. The AI-ERA approach consists of two modes, namely offline and online modes. First, a deep neural network (DNN) model is trained with a dataset generated at ns-3 in the offline mode. Second, the proposed AI-ERA approach utilizes the pre-trained DNN model in the online mode to proactively assign an efficient SF for the end device before each uplink packet transmission.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See the published paper, Link: https://ieeexplore.ieee.org/abstract/document/10050828, for detailed instructions on how to deploy the project on a live LoRaWAN network in ns-3.

### Prerequisites

The code provided is designed to be compatible with PyTorch 1.7.1 and Python 3.8, ensuring optimal functionality and performance. PyTorch 1.7.1 is a powerful deep-learning framework that offers a wide range of tools and capabilities for building and training neural networks. Python 3.8, on the other hand, provides a robust and versatile programming environment. By leveraging the features of both PyTorch 1.7.1 and Python 3.8, the code can take advantage of the latest advancements in deep learning and harness the flexibility and ease of use offered by Python. The code is also compatible with ns3.33, a popular network simulation framework, allowing seamless integration with network simulations and enabling comprehensive analysis and evaluation of network performance.

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

The main file comprises MLP code for 5 layers.

```
[your_env_path] python3 main_code.py
```

## Dataset

To understand the dataset generation and labeling, please refer to https://ieeexplore.ieee.org/abstract/document/10050828. The dataset is in the ''data'' folder. Currently, the Python code inside the folder can generate a sequence of 6 groups (24 flattened features) required for MLP.

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
