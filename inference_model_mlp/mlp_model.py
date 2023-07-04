import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, batch_size, feature, device, dropout=0.5):
        super(MLP, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.dropout = dropout
        #self.group = group
        self.feature = feature
    
        self.fc_1 = nn.Linear(self.feature*6, 50)
        self.fc_2 = nn.Linear(50, 100)
        self.fc_3 = nn.Linear(100, 150)
        self.fc_4 = nn.Linear(150, 200)
        self.fc_5 = nn.Linear(200, 6)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, self.feature*6)
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)
        x = self.relu(x)
        x = self.fc_3(x)
        x = self.relu(x)
        x = self.fc_4(x)
        x = self.relu(x)
        x = self.fc_5(x)

        return x
