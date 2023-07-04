# AI-ERA
To understand this code, please refer to our paper titled "AI-ERA: Artificial Intelligence-Empowered Resource Allocation for LoRa-Enabled IoT Applications," Link: https://ieeexplore.ieee.org/abstract/document/10050828.

A. Farhad and J. -Y. Pyun, "AI-ERA: Artificial Intelligence-Empowered Resource Allocation for LoRa-Enabled IoT Applications," in IEEE Transactions on Industrial Informatics, doi: 10.1109/TII.2023.3248074.


CODE DESCRIPTION:

'main_code.py'

This file comprises an MLP code of 5 fully connected layers.

'data'

This folder contains the original dataset, where the 'dataset_gen' file is used to generate the sequence required for the MLP code.

'model'

This folder contains the saved MLP model.

'inference_mode_mlp'

This folder contains the inference model, which is executed from the ns3.33 (lorawan module).
The model gets a new input from the ns-3 lorawan-module and classifies a suitable spreading factor (SF) allocated to the end device during network simulation.

Authors: Arshad Farhad, Ph.D.
