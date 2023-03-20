import torch, pickle
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as modelss
# import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import torch.nn.functional as F
from utils import load_data_TITS_transform, load_data_TITS_notransform
from models import *
import numpy as np
import torch.utils.data
# from torchsampler import ImbalancedDatasetSampler
from sampler import BalancedBatchSampler
from torchmetrics.classification import ConfusionMatrix
import matplotlib.pyplot as plt
# from sam import SAM
# from cosine_annealing_warmup import CosineAnnealingWarmupRestarts


import os
# os.environ["CUDA_VISIBLE_DEVICES"]="5"
torch.set_num_threads(3)
torch.cuda.empty_cache()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 30
batch_size = 64
learning_rate = 0.0005
weight_delay = 0.01
pixel_size = 40
Balance = True
input_channel = 5
num_classses = 6
kernel_size = 7
transformation = False
rescale = True

Mode_Index = {"walk": 0,  "bike": 1,  "car": 2, "taxi": 2, "bus": 3, "subway": 4, "train": 5}
classes = ('walk', 'bike', 'car&taxi', 'bus', 'subway', 'train')

print(f'parameters. pixel size: {pixel_size}, num epoches: {num_epochs}, batch_size: {batch_size}, learning rate: {learning_rate}, weight delay: {weight_delay}; channel: {input_channel}; kernel size: {kernel_size}; transformation: {transformation}')

print('--> data load...')
filename = '/home/xieyuan/Transportation-mode/Traj2Image/datafiles/Geolife/trips_traj2image_trip_shift_rescale_%dclass_pixelsize%d_back0_180bearing_unfixed_TITS_5s.pickle'%(num_classses, pixel_size)
if transformation:
    train_dataset, test_dataset, train_x_label, test_y_geolife, train_scale_geolife, test_scale_geolife = load_data_TITS_transform(filename, input_channel=input_channel)
else:
    train_dataset, test_dataset, train_x_label, test_y_geolife, train_scale_geolife, test_scale_geolife = load_data_TITS_notransform(filename, input_channel=input_channel)


print(train_x_label.shape, type(train_x_label))
print(test_y_geolife.shape, type(test_y_geolife))

if Balance == True:
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=BalancedBatchSampler(train_dataset, torch.from_numpy(train_x_label)), batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
elif Balance == False:
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


class_dict_train={}
class_scale_train = defaultdict(list)
for y in range(len(train_x_label)):
    if train_x_label[y] not in class_dict_train:
        class_dict_train[train_x_label[y]]=1
        class_scale_train[train_x_label[y]].append(train_scale_geolife[y][0])
    else:
        class_dict_train[train_x_label[y]]+=1
        class_scale_train[train_x_label[y]].append(train_scale_geolife[y][0])     
class_dict_train = sorted(class_dict_train.items(), key=lambda item:item[0])
class_dict_train = dict(class_dict_train)
print('Original Train geolife class:', class_dict_train, class_dict_train.values(), np.array(list(class_dict_train.values())).sum())
class_dict_test={}
for y in range(len(test_y_geolife)):
    if test_y_geolife[y] not in class_dict_test:
        class_dict_test[test_y_geolife[y]]=1
        class_scale_train[train_x_label[y]].append(train_scale_geolife[y][0])
    else:
        class_dict_test[test_y_geolife[y]]+=1
        class_scale_train[train_x_label[y]].append(train_scale_geolife[y][0])
class_dict_test = sorted(class_dict_test.items(), key=lambda item:item[0])
class_dict_test = dict(class_dict_test)
print('Original Test geolife class:', class_dict_test, class_dict_test.values(), np.array(list(class_dict_test.values())).sum())

import pandas as pd
# scale distribution
# all_scale = []
# for i in class_scale_train.keys():
#     for k in class_scale_train[i]:
#         all_scale.append(k)
# print('scale value of mode all', pd.Series(all_scale).describe(percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
#                                                                                             0.7, 0.8, 0.9, 0.99, 1]))
# df = pd.Series(all_scale)
# label_list = [0, 0.001, 0.005, 0.008, 0.009, 0.01, 0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 1899]
# label_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 1899]
# label_list = [0, 0.001, 0.005, 0.008, 0.009, 0.01]
# label_list = [9, 10, 20, 30, 50, 100, 1899]
# ax = pd.cut(df, bins=label_list)
# values = ax.value_counts(sort=False).values
# labels = [str(label_list[i]) + '-' + str(label_list[i+1]) for i in range(len(label_list)-1)]
# df = pd.DataFrame(values, index=labels)
# res = df.plot(kind='bar', legend = False).get_figure()
# img_path = 'plot_all.png'
# res.savefig(img_path)
# import pdb; pdb.set_trace()

fig = plt.figure()
for i in class_scale_train.keys():
    print('scale value of mode %s'%classes[i], pd.Series(class_scale_train[i]).describe(percentiles=[0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                                                                                            0.7, 0.8, 0.9, 0.99, 0.999, 0.9999, 1]))
    df = pd.Series(class_scale_train[i])
    label_list = [0, 0.01, 0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 1673]
    ax = pd.cut(df, bins=label_list)
    
    values = ax.value_counts(sort=False).values
    labels = [str(label_list[i]) + '-' + str(label_list[i+1]) for i in range(len(label_list)-1)]

    df = pd.DataFrame(values, index=labels)
    res = df.plot(kind='bar', legend = False).get_figure()

    img_path = 'plot_%s.png'%classes[i]
    res.savefig(img_path)

import pdb; pdb.set_trace()

print('--> Build model...')

print('ConvNet_Multilayer')
model = MultiLayerConvNet(input_channel, num_classses, kernel_size).to(device)

# print('Pre-trained ResNet18')
# model_e = modelss.resnet18(pretrained=True).to(device)   # resnet18
# torch.save(model_e.state_dict(), 'resnet18.pkl')
# model = ResNet18().to(device)   # resnet18
# model_dict = torch.load('resnet18.pkl')
# model_dict.pop('conv1.weight', None)
# model.load_state_dict(model_dict, strict=False)

# print('ResNet18')
# model = ResNet18().to(device)   # resnet18

# print('ResNet50')
# model = ResNet50().to(device)   # resnet18

model = model.cuda()
# print(next(model.parameters()).device)

# loss and optimize
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_delay)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.01)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=weight_delay) # lr is min lr
# scheduler = CosineAnnealingWarmupRestarts(optimizer,
#                                           first_cycle_steps=200,
#                                           cycle_mult=1.0,
#                                           max_lr=0.1,
#                                           min_lr=0.001,
#                                           warmup_steps=50,
#                                           gamma=1.0)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)

# training loop
loss_list = []
n_total_steps = len(train_loader)
train_acc = []
test_acc = []

# import pdb; pdb.set_trace()
print('--> Model training...', 'epoch:', num_epochs)

best_acc = 0
best_epoch = 0

# weights = torch.Tensor(list(class_dict_train.values()))
# print(weights)

if rescale:
    train_loss = []
    for epoch in range(num_epochs):

        model.train()

        n_correct = 0
        n_samples = 0
        losses = []
        epoch_loss = 0

        # training phase
        for i, (images, labels, scale) in enumerate(train_loader):

            # images = images.reshape(-1, 128*128).to(device)
            images = images.to(device)
            labels = labels.to(device)
            if transformation:
                scale = scale.unsqueeze(-1)
                scale = scale.to(device)
                scale = scale.float()
            else:
                scale = scale.to(device)

            # forward
            outputs = model(images, scale)
            loss = criterion(outputs, labels)
            # print(outputs)

            # loss = weights[labels] * loss
            # loss = loss.mean()

            # backward
            optimizer.zero_grad()
            loss.backward()

            # print(i, loss.item())

            # import pdb; pdb.set_trace()
            optimizer.step()

            _, predictions = torch.max(outputs, 1)
            n_samples += labels.shape[0]
            n_correct += (predictions == labels).sum().item()

            losses.append(loss.item())
            epoch_loss += loss.item()
        # scheduler.step()
        acc_train = 100.0 * n_correct / n_samples
        epoch_loss = epoch_loss / len(train_loader)
        print(f'epoch {epoch+1} / {num_epochs}, loss = {epoch_loss}, acc = {acc_train:.4f}')  
        train_loss.append(epoch_loss)
        train_acc.append(acc_train)

        # validation phase
        n_correct = 0
        n_samples = 0

        n_class_correct = [0 for i in range(num_classses)]
        n_class_samples = [0 for i in range(num_classses)]

        # n_class_correct = [0 for i in range(4)]
        # n_class_samples = [0 for i in range(4)]

        test_results = []
        label_list = []
        pred_list = []

        model.eval()
        for images, labels, scale in test_loader:

            images = images.to(device)
            labels = labels.to(device)
            if transformation:
                scale = scale.unsqueeze(-1)
                scale = scale.to(device)
                scale = scale.float()
            else:
                scale = scale.to(device)
            
            outputs = model(images, scale)

            # label_list += labels.tolist()
            # pred_list += 

            # value, index
            _, predictions = torch.max(outputs, 1)
            n_samples += labels.shape[0]
            n_correct += (predictions == labels).sum().item()
            test_results.append(predictions.tolist())
            # print(predictions.tolist())
            # print(labels.tolist())
            predictions = predictions.tolist()
            labels = labels.tolist()
            # print(len(labels), labels)
        
            for i in range(len(labels)):
                # print(i, labels[i])
                label = labels[i]
                pred = predictions[i]
                if (label == pred):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1
                label_list.append(label)
                pred_list.append(pred)

        acc_val = 100.0 * n_correct / n_samples
        for i in range(num_classses):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'Accuracy of {classes[i]}: {acc:.4f} %')
        print(f'test acc = {acc_val:.4f}')

        # print(pred_list)
        confmat = ConfusionMatrix(task="multiclass", num_classes=num_classses)
        print(confmat(torch.tensor(pred_list), torch.tensor(label_list)))


        test_acc.append(acc_val)

        if acc_val > best_acc:
            best_epoch=epoch+1
            best_acc = max(acc_val, best_acc)
else:
    train_loss = []
    for epoch in range(num_epochs):

        model.train()

        n_correct = 0
        n_samples = 0
        losses = []
        epoch_loss = 0

        # training phase
        for i, (images, labels) in enumerate(train_loader):

            # images = images.reshape(-1, 128*128).to(device)
            images = images.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(images)
            loss = criterion(outputs, labels)
            # print(outputs)

            # backward
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            _, predictions = torch.max(outputs, 1)
            n_samples += labels.shape[0]
            n_correct += (predictions == labels).sum().item()

            losses.append(loss.item())
            epoch_loss += loss.item()
        # scheduler.step()
        acc_train = 100.0 * n_correct / n_samples
        epoch_loss = epoch_loss / len(train_loader)
        print(f'epoch {epoch+1} / {num_epochs}, loss = {epoch_loss}, acc = {acc_train:.4f}')  
        train_loss.append(epoch_loss)
        train_acc.append(acc_train)

        # validation phase
        n_correct = 0
        n_samples = 0

        n_class_correct = [0 for i in range(num_classses)]
        n_class_samples = [0 for i in range(num_classses)]

        # n_class_correct = [0 for i in range(4)]
        # n_class_samples = [0 for i in range(4)]

        test_results = []
        label_list = []
        pred_list = []

        model.eval()
        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)

            # label_list += labels.tolist()
            # pred_list += 

            # value, index
            _, predictions = torch.max(outputs, 1)
            n_samples += labels.shape[0]
            n_correct += (predictions == labels).sum().item()
            test_results.append(predictions.tolist())
            # print(predictions.tolist())
            # print(labels.tolist())
            predictions = predictions.tolist()
            labels = labels.tolist()
            # print(len(labels), labels)
        
            for i in range(len(labels)):
                # print(i, labels[i])
                label = labels[i]
                pred = predictions[i]
                if (label == pred):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1
                label_list.append(label)
                pred_list.append(pred)

        acc_val = 100.0 * n_correct / n_samples
        for i in range(num_classses):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'Accuracy of {classes[i]}: {acc:.4f} %')
        print(f'test acc = {acc_val:.4f}')

        # print(pred_list)
        confmat = ConfusionMatrix(task="multiclass", num_classes=num_classses)
        print(confmat(torch.tensor(pred_list), torch.tensor(label_list)))


        test_acc.append(acc_val)

        if acc_val > best_acc:
            best_epoch=epoch+1
            best_acc = max(acc_val, best_acc)




print("best_acc = {:3.1f}({:d})".format(best_acc, best_epoch))

print('loss list:', train_loss)
print('train acc:', train_acc)
print('test acc:', test_acc)
