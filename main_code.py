import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
from ptflops import get_model_complexity_info

#import onnx

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

print('pytorch :', torch.__version__)
print('graphic name :', torch.cuda.get_device_name())


### parameters

retrain = False
#checkpoint_name = '[MLP](epoch-10000)-(init_lr-0.0001)-(batch-256)-(layer-5)-(acc-82).pt'

# parameters for dataset
random_seed = 42
batch_size = 256
shuffle = True
drop_last = True

# parameters for train
r_epoch = 0
lr = 0.0001  # optimizer : learning rate
weight_decay = 0.000001  # optimizer
num_epoch = 10  # epoch
hidden_unit = 200  # MLP hidden unit
num_layers = 1  # MLP layer (>1)
drop_out = 0.5  # MLP dropout

# parameters for check
view_train_iter = 100
view_val_iter = 10  # view_train_iter// 50
save_point = 0.9  #

### random seed (torch, numpy, random)

def torch_random_seed(on_seed=False, random_seed=42):
    if on_seed:
        torch.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        np.random.seed(random_seed)
        random.seed(random_seed)


torch_random_seed(on_seed=True, random_seed=random_seed)

### evaluation (accuracy, precision recall, f1 score)

def get_clf_eval(y_true, y_pred, average='weighted'):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, warn_for=tuple())
    return accuracy, precision, recall, f1


## generate dataset
# 3.1. load dataset
# 3.2. split data (train data, validation data, test data)
# 3.3. convert numpy to tensor 

name = 'mlp_6_3'
x_data = np.load('./data/x_dat_6_3.npy')
y_data = np.load('./data/y_dat_6_3.npy')
y_data = y_data-7

print(x_data.shape)
print(y_data.shape)

### split data (train data, validation data, test data)
x_train, x_test, y_train, y_test = train_test_split(x_data,
                                                    y_data,
                                                    test_size=0.1,
                                                    random_state=random_seed,
                                                    stratify=y_data)
#x_train, x_val, y_train, y_val = train_test_split(x_train,
#                                                  y_train,
#                                                  test_size=0.5,
#                                                  random_state=random_seed,
#
#  stratify=y_train)
x_val = x_test
y_val = y_test


print("x_train shape :", x_train.shape, y_train.shape)
print("x_val shape :", x_val.shape, y_val.shape)
print("x_test shape :", x_test.shape, y_test.shape)

y_data_count = [len(y_data[y_data==i]) for i in np.unique(y_data)]
y_train_count = [len(y_train[y_train==i]) for i in np.unique(y_train)]
y_val_count = [len(y_val[y_val==i]) for i in np.unique(y_val)]
y_test_count = [len(y_test[y_test==i]) for i in np.unique(y_test)]

print("all SF(0-5) ratio :", y_data_count)
print("*"*65)
print("train SF(0-5) ratio :", y_train_count)
print("valid SF(0-5) ratio :", y_val_count)
print("test SF(0-5) ratio :", y_test_count)

### convert numpy to tensor


def generate_tensor_loader(x_data, y_data, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last):
    x_tensor = torch.tensor(x_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.long).view(-1)

    return DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

trainloader = generate_tensor_loader(x_train, y_train, batch_size=batch_size, shuffle=True, drop_last=True)
validationloader = generate_tensor_loader(x_val, y_val, batch_size=batch_size, shuffle=False, drop_last=True)
testloader = generate_tensor_loader(x_val, y_val, batch_size=batch_size, shuffle=False, drop_last=True)
testloader = generate_tensor_loader(x_val, y_val, batch_size=y_test.shape[0], shuffle=False, drop_last=True)
# testloader = generate_tensor_loader(x_val, y_val, batch_size=1, shuffle=False, drop_last=True)

for x, label in trainloader:
    print(x.shape, label.shape)
    break


## Model
# 4.1. DNN Model
# 4.2. Loss/Cost Function, Optimizer, GPU Device
# 4.3. Transfer Learning

class MLP(nn.Module):
    def __init__(self, batch_size, group, feature, device, dropout=0.5):
        super(MLP, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.dropout = dropout
        self.group = group
        self.feature = feature
    
        self.fc_1 = nn.Linear(self.feature*self.group, 50)
        self.fc_2 = nn.Linear(50, 100)
        self.fc_3 = nn.Linear(100, 150)
        self.fc_4 = nn.Linear(150, 200)
        self.fc_5 = nn.Linear(200, 6)
        #self.fc_6 = nn.Linear(250, 300)
        #self.fc_7 = nn.Linear(300, 350)
        #self.fc_8 = nn.Linear(350, 400)
        #self.fc_9 = nn.Linear(400, 450)
        #self.fc_10 = nn.Linear(450, 6)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, self.feature*self.group)
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)
        x = self.relu(x)
        x = self.fc_3(x)
        x = self.relu(x)
        x = self.fc_4(x)
        x = self.relu(x)
        x = self.fc_5(x)

        #x = self.relu(x)
        #x = self.fc_6(x)
        #x = self.relu(x)
        #x = self.fc_7(x)
        #x = self.relu(x)
        #x = self.fc_8(x)
        #x = self.relu(x)
        #x = self.fc_9(x)
        #x = self.relu(x)
        #x = self.fc_10(x)

        return x
    
    ### Loss/Cost Function, Optimizer, GPU Device

# gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)


# MLP
model = MLP(batch_size=batch_size,
             group=x_train.shape[1],
             feature=x_train.shape[2],
             device=device,
             ).to(device)


# CrossEntropy
loss_function = nn.CrossEntropyLoss()
print('model parameters',model)
#from torchsummary import summary
#from torchvision import models
from torchsummary import summary
##model = model.vgg16()
#print('params',model)

summary(model, (256, 6, 4))
# Adam
optimizer = optim.Adam(model.parameters(), lr=lr,
                       weight_decay=weight_decay)  # optimizer
# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# Scheduler
lr_scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer=optimizer,
    milestones=[int(num_epoch * 0.5),
                int(num_epoch * 0.75)],
    gamma=0.1,
    last_epoch=-1)

if retrain:
    checkpoint_path = './model/' + checkpoint_name
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    r_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(checkpoint_path)



from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model, (256, 6, 4), as_strings=True, verbose=True)
        print("\nMACS: {}, PARAMS: {}".format(macs, params))
    return total_params
    
count_parameters(model)
## Train/Validation/Test
# 5.1. Train/Validation
# 5.2. Plot Train/Validation figure
# 5.3. Evaluation
# 5.4. Confusion Matrix

### Train/Validation

correct = 0
total = 0
tmp_acc = 0
train_acc = []
loss_arr = []
val_view_acc = []
start = time.time()  # 시작 시간 저장

for epoch in range(num_epoch):
    epoch = epoch + r_epoch + 1
    for train_iter, (train_x, train_y_true) in enumerate(trainloader):
        model.train()  # Train mode
        model.zero_grad()  # model zero initialize
        optimizer.zero_grad()  # optimizer zero initialize
        
        train_x, train_y_true = train_x.to(device), train_y_true.to(device)  # device(gpu)
        train_y_pred = model.forward(train_x)  # forward
        
        loss = loss_function(train_y_pred, train_y_true)  # loss function
        loss.backward()  # backward
        optimizer.step()  # optimizer
        _, pred_index = torch.max(train_y_pred, 1)
    
        if train_iter % view_train_iter == 0:
            loss_arr.append(loss.item())
            total += train_y_true.size(0)  # y.size(0)
            correct += (pred_index == train_y_true).sum().float()  # correct
            tmp_acc = correct / total  # accuracy
            train_acc.append(tmp_acc.tolist())
            print("[Train] ({}, {}) loss = {:.5f}, Accuracy = {:.4f}, lr={:.6f}".format(epoch, train_iter, loss.item(), tmp_acc, optimizer.param_groups[0]['lr']))
    lr_scheduler.step()    
    # validation 
    if epoch % view_val_iter == 0: 
        val_actual_tmp, val_pred_tmp = [], []
        for val_iter, (val_x, val_y_true) in enumerate(validationloader):
            model.eval()
            val_x, val_y_true = val_x.to(device), val_y_true.to(device)  # device(gpu)
            val_y_pred = model.forward(val_x)  # forward
            _, val_pred_index = torch.max(val_y_pred, 1)
            
            val_pred_index_cpu = val_pred_index.cpu().detach().numpy()
            val_y_true_cpu = val_y_true.cpu().detach().numpy()
            
            val_actual_tmp.append(val_pred_index_cpu.tolist())
            val_pred_tmp.append(val_y_true_cpu.tolist())
            
        val_acc, val_precision, val_recall, val_f1 = get_clf_eval(val_actual_tmp[1], val_pred_tmp[1])
        val_view_acc.append(val_acc)
        print("*[Valid] Accuracy:{:.4f}, Precison:{:.4f}, Recall:{:.4f}, F1 Score:{:.4f}".format(
             val_acc, val_precision, val_recall, val_f1))
print("Time : ", time.time()-start,'[s]',sep='')

### Plot Train/Validation figure

f = plt.figure(figsize=[8, 5])
f.set_facecolor("white")

#loss_plt = plt.subplot(2,1,1)
#acc_plt = plt.subplot(2,1,2)
# print(val_view_acc)
print(len(val_view_acc))
# print(np.arange(10, epoch+1, view_val_iter).shape)
plt.style.use(['default'])

# plt.title("Train Result", fontsize=12)
plt.plot(loss_arr)
plt.plot(train_acc)

try:
    plt.scatter(np.arange(10, epoch+1, view_val_iter), val_view_acc, c='limegreen', s=10)
except:
    plt.scatter(np.arange(10, epoch, view_val_iter), val_view_acc, c='limegreen', s=10)

plt.legend(['Loss', 'Accuracy','Validation'], fontsize=16)
#plt.set_xticklabels(labels, fontsize=15)
#plt.set_yticklabels(labels, fontsize=15)
plt.xticks(fontsize= 15)
plt.yticks(fontsize= 15)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Normalized Probability", fontsize=16)
plt.ylim((0.0, 1.0))
plt.grid(True)
plt.show()
print("./plot/"+"[plot] "+model._get_name()+"_"+'layer'+str(num_layers)+".pdf")
f.savefig("./plot/"+"[plot] "+model._get_name()+"_"+'layer'+str(num_layers)+".pdf")


### Evaluation

# for confusion matrix

test_pred_acc = []  
test_actual_acc = []  
start = time.time()  # 시작 시간 저장
device = 'cpu'
model.to(device)

model.eval()
with torch.no_grad():
    test_acc_tmp, test_precision_tmp, test_recall_tmp, test_f1_tmp = [], [], [], []
    for test_iter, (test_x, test_y_true) in enumerate(testloader):
        test_x, test_y_true = test_x.to(device), test_y_true.to(device)
        test_y_pred = model.forward(test_x)  # forward
        
        _, test_pred_index = torch.max(test_y_pred, 1)
        
        test_pred_index_cpu = test_pred_index.cpu().detach().numpy()
        test_y_true_cpu = test_y_true.cpu().detach().numpy()
        
        test_pred_acc.append(test_pred_index_cpu)
        test_actual_acc.append(test_y_true_cpu)
            
        test_acc, test_precision, test_recall, test_f1 = get_clf_eval(test_y_true_cpu, test_pred_index_cpu)

        test_acc_tmp.append(test_acc), test_precision_tmp.append(test_precision), test_recall_tmp.append(test_recall), test_f1_tmp.append(test_f1)
    test_acc_mean = sum(test_acc_tmp, 0.0)/len(test_acc_tmp)
    test_precision_mean = sum(test_precision_tmp, 0.0)/len(test_precision_tmp)
    test_recall_mean = sum(test_recall_tmp, 0.0)/len(test_recall_tmp)
    test_f1_mean = sum(test_f1_tmp, 0.0)/len(test_f1_tmp)
    print("[Test] Accuracy : {:.4f}, Precision : {:.4f}, Recall : {:.4f}, F1 Score : {:.4f}".format(
         test_acc_mean, test_precision_mean, test_recall_mean, test_f1_mean))
    print("[Test] Model Performance : {:.5f}".format(test_acc_mean))
print("Time : ", time.time()-start,'[s]',sep='')

### Confusion Matrix


def plot_confusion_matrix(data, labels):
    """Plot confusion matrix using heatmap.
    data (list of list): List of lists with confusion matrix data.
    labels (list): Labels which will be plotted across x and y axis.
    output_filename (str): Path to output file.
    """
    sns.set(color_codes=True)
    #plt.figure(1, figsize=(8, 5))
    plt.figure(figsize = (10,7))
    # plt.title("Confusion Matrix", fontsize=20)
    #sns.set(font_scale=1.2)
    ax = sns.heatmap(data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'}, fmt='.3f')
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_yticklabels(labels, fontsize=14)
    ax.set_ylabel("True Label", fontsize=15)
    ax.set_xlabel("Predicted Label", fontsize=15)
    ax.figure.axes[-1].yaxis.label.set_size(15)
    print("./plot/"+"[MLP_mat]"+model._get_name()+"_"+'layer'+str(num_layers)+".pdf")
    plt.savefig("./plot/"+"[MLP_mat] "+model._get_name()+"_"+'layer'+str(num_layers)+".pdf", bbox_inches='tight', dpi=300)
    # plt.close()
    
    import sklearn

test_pred_acc = np.array(test_pred_acc).reshape(-1)
test_actual_acc = np.array(test_actual_acc).reshape(-1)

# test_pred_acc = test_pred_acc.numpy()
# test_actual_acc = test_actual_acc.numpy()

cf = confusion_matrix(y_true=test_actual_acc, y_pred=test_pred_acc, normalize = 'true')
cf_norm= cf / cf.astype(np.float).sum(axis=1)

plot_confusion_matrix(cf_norm, ['SF7', 'SF8', 'SF9', 'SF10', 'SF11', 'SF12'])

### test confusion matrix

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
import matplotlib.pyplot as plt
con_mat = confusion_matrix(y_true=test_actual_acc,y_pred=test_pred_acc, normalize = 'true')
print(f'Y_True: {test_actual_acc.shape}\nY_pred: {test_pred_acc.shape}.')
plt.figure(figsize = (10,7))
# sn.heatmap(con_mat, annot=True, fmt='.2f')
sn.heatmap(con_mat, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'}, fmt='.3f')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
#plt.ylabel('Truth')
plt.savefig("./plot/"+"[MLP] "+model._get_name()+"_"+'layer'+str(num_layers)+".pdf", bbox_inches='tight', dpi=300)



### test finished confsion matrix

## Save Model
# 6.1. Save .pt
# 6.2. Save .onnx

epoch_str = str(epoch)
lr_str = str(lr)
batch_str= str(batch_size)
acc_str= str(int(test_acc_mean*100))

model_name = "["+model._get_name()+"]"+"(MLPepoch-"+epoch_str+")-"+"(init_lr-"+lr_str+")-"+"(batch-"+batch_str+")-"+"("+'layer-'+str(num_layers)+")-"+"(acc-"+acc_str+").pt"
save_path = './model/LSTM/' + model_name
print(save_path)

torch.save({'epoch' : epoch,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'loss' : loss
           }, save_path)

### Save .onnx

dummy_input = torch.randn(1, 6, 3, device=device)

# model_name_onnx = "["+model._get_name()+"]"+"(epoch-"+epoch_str+")-"+"(init_lr-"+lr_str+")-"+"(batch-"+batch_str+")-"+"(acc-"+acc_str+").onnx"
# save_path_onnx = './model/' + model_name_onnx

# torch.onnx.export(model, dummy_input, save_path_onnx, verbose=True)
