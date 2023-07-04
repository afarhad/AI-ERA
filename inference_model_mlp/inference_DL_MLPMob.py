import torch
from mlp_model import MLP
import numpy as np
import pickle


# load input data
_file = 'input.txt'
_dir = '/home/trust-server/바탕화면/DL/MLPMob/MLP_2/data/'
path = np.loadtxt(_dir + _file)

# generate sequence data
def generate_sequence_dataset(x_data, sequence=6):
    x_data_list = []
    x_data_length = len(x_data)
    for i in range(x_data_length-sequence):
        x_data_list.append(x_data[i:i+sequence])
    return np.array(x_data_list)

x_data = generate_sequence_dataset(path)

# convert np into tensor 
x_tensor = torch.tensor(x_data, dtype=torch.float32)

# device 
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'


# gru model
# arshad change num_layers. 
model = MLP(batch_size=1,
             feature=4,
             device=device).to(device)

# .pt checkpoint
checkpoint_path = '/home/trust-server/바탕화면/DL/MLPMob/MLP_2/model/' + 'mlp_6_3_acc82_epoch10000.pt'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])

# .pt inference
model.eval()
with torch.no_grad():
    for data in x_tensor:
        data = data.reshape(1, x_data.shape[1], x_data.shape[2])
        #data = data.to(device)
        data = data.reshape(1, 6, 4)
        predicted_label = model.forward(data)
        _, test_pred_index = torch.max(predicted_label, 1)
        pred = test_pred_index.cpu().detach().numpy()[0] + 6
        print("SF: ", pred)
        # 'w' means clean the file before writing
        f = open('/home/trust-server/바탕화면/DL/MLPMob/MLP_2/data/predictSF.txt', 'w')
        f.write(str(pred))
        f.close()



