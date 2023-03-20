import sys
import os.path as osp
import time
# from PIL import Image
import numpy as np
import pickle
from torch.utils.data import TensorDataset, DataLoader
import torch
from torch.utils.data import Dataset
# import kornia as K
import torchvision.transforms as transforms
import torchvision
# import matplotlib.pyplot as plt
# from PIL.Image import Image
# import mpl_toolkits.axes_grid1 as axes_grid1

Mode_Index = {0: "walk",  1: "bike",  2: "car", 2: "taxi", 3: "bus", 4: "subway", 5: "train"}
interplotion = True
num_class = 6
visualize = False

def load_data_TITS_notransform(filename, input_channel):

    with open(filename, 'rb') as f:
        kfold_dataset = pickle.load(f)
    dataset = kfold_dataset
    
    train_x_geolife = np.array(dataset[0 + input_channel])
    train_y_geolife = np.array(dataset[7])
    train_scale_geolife_init = np.array(dataset[8])
    
    test_x_geolife = np.array(dataset[9 + input_channel])
    test_y_geolife = np.array(dataset[16])
    test_scale_geolife_init = np.array(dataset[17])

    train_scale_geolife = []
    for i in train_scale_geolife_init:
        train_scale_geolife.append([i])
    train_scale_geolife = np.array(train_scale_geolife)

    test_scale_geolife = []
    for i in test_scale_geolife_init:
        test_scale_geolife.append([i])
    test_scale_geolife = np.array(test_scale_geolife)

    train_dataset_geolife = TensorDataset(
        torch.from_numpy(train_x_geolife).to(torch.float),
        torch.from_numpy(train_y_geolife),
        torch.from_numpy(train_scale_geolife).to(torch.float)
    )
    test_dataset_geolife = TensorDataset(
        torch.from_numpy(test_x_geolife).to(torch.float),
        torch.from_numpy(test_y_geolife),
        torch.from_numpy(test_scale_geolife).to(torch.float)
    )
    print(f'train: {train_x_geolife.shape}; {type(train_x_geolife)}')
    print(f'train label: {train_y_geolife.shape}, {type(train_y_geolife)}')
    print(f'test: {test_x_geolife.shape}; {type(test_x_geolife)}')
    print(f'test label: {test_y_geolife.shape}; {type(test_y_geolife)}')
    print(f'train_data: {type(train_dataset_geolife)}, {type(test_dataset_geolife)}')

    return train_dataset_geolife, test_dataset_geolife, train_y_geolife, test_y_geolife, train_scale_geolife, test_scale_geolife


def load_data_TITS_transform(file_path, input_channel):

    with open(file_path, 'rb') as f:
        kfold_dataset = pickle.load(f)
    dataset = kfold_dataset

    train_x_geolife = np.array(dataset[0 + input_channel])
    train_y_geolife = np.array(dataset[7])
    train_scale_geolife_init = np.array(dataset[8])
    
    test_x_geolife = np.array(dataset[9 + input_channel])
    test_y_geolife = np.array(dataset[16])
    test_scale_geolife_init = np.array(dataset[17])

    train_scale_geolife = []
    for i in train_scale_geolife_init:
        train_scale_geolife.append([i])
    train_scale_geolife = np.array(train_scale_geolife)

    test_scale_geolife = []
    for i in test_scale_geolife_init:
        test_scale_geolife.append([i])
    test_scale_geolife = np.array(test_scale_geolife)

    train_composed = torchvision.transforms.Compose([ToTensor(),
                                                    #  torchvision.transforms.CenterCrop((224, 224)),
                                                     torchvision.transforms.RandomRotation(90),
                                                     torchvision.transforms.RandomHorizontalFlip(),
                                                     torchvision.transforms.RandomResizedCrop(224, scale=(0.8, 1.2)),
                                                    ])
    test_composed = torchvision.transforms.Compose([ToTensor(),
                                                     torchvision.transforms.Resize((224, 224)),
                                                    ])
    train_dataset_geolife = TrainDataset(file_path, pixel_size=pixel_size, input_channel=input_channel, transform=train_composed)
    test_dataset_geolife = TestDataset(file_path, pixel_size=pixel_size, input_channel=input_channel, transform=test_composed)

    train_scale_geolife = TensorDataset(
        torch.from_numpy(train_scale_geolife).to(torch.float)
    )
    test_scale_geolife = TensorDataset(
        torch.from_numpy(test_scale_geolife).to(torch.float)
    )
    
    print(f'train label: {train_y_geolife.shape}, {type(train_y_geolife)}')
    print(f'test: {test_y_geolife.shape}; {type(test_y_geolife)}')
    print(f'train_data: {len(train_dataset_geolife)}, {type(train_dataset_geolife)}')
    print(f'test_data: {len(test_dataset_geolife)}, {type(test_dataset_geolife)}')

    return train_dataset_geolife, test_dataset_geolife, train_y_geolife, test_y_geolife, train_scale_geolife, test_scale_geolife


class TrainDataset(Dataset):

    def __init__(self, file_path, pixel_size, input_channel, transform=None):

        self.pixel_size = pixel_size
        self.input_channel = input_channel
        self.transform = transform
        
        num_class = 6
        # filename = '/home/xieyuan/Transportation-mode/Traj2Image/datafiles/Geolife/trips_traj2image_trip_shift_rescale_%dclass_pixelsize%d_back0_180bearing_unfixed_TITS.pickle'%(num_class, self.pixel_size)
        with open(file_path, 'rb') as f: 
            kfold_dataset = pickle.load(f)
        dataset = kfold_dataset
        
        self.train_x_geolife = np.array(dataset[0 + input_channel])
        self.train_y_geolife = np.array(dataset[7])
        self.train_scale_geolife = np.array(dataset[8])
        self.n_samples = self.train_x_geolife.shape[0]

    def __getitem__(self, index):
        sample = self.train_x_geolife[index]
        label = self.train_y_geolife[index]
        scale = self.train_scale_geolife[index]

       # speed & count & acc & bearing & bearing rate
        if visualize:
            count = 0
            img = sample
            image_count = 0
            fig = plt.figure()
            fig.set_facecolor('white')
            grid_shift = axes_grid1.AxesGrid(
            fig, 111, nrows_ncols=(1, self.input_channel), axes_pad = 0.5, cbar_location = "right",
            cbar_mode="each", cbar_size="15%", cbar_pad="5%",)
            speed_array_final = img[0]       
            count_final = img[1]
            acc_final = img[2]    
            bearing_array_final = img[3] 
            bearing_rate_array_final = img[4]
            
            im0 = grid_shift[image_count].imshow(speed_array_final, cmap='jet', interpolation='nearest')
            grid_shift.cbar_axes[0].colorbar(im0)

            im1 = grid_shift[image_count+1].imshow(count_final, cmap='jet', interpolation='nearest')
            grid_shift.cbar_axes[1].colorbar(im1)

            im2 = grid_shift[image_count+2].imshow(acc_final, cmap='jet', interpolation='nearest')
            grid_shift.cbar_axes[2].colorbar(im2)

            im3 = grid_shift[image_count+3].imshow(bearing_array_final, cmap='jet', interpolation='nearest')
            grid_shift.cbar_axes[3].colorbar(im3)

            im4 = grid_shift[image_count+4].imshow(bearing_rate_array_final, cmap='jet', interpolation='nearest')
            grid_shift.cbar_axes[4].colorbar(im4)

            img_path = 'image_class_fixed_pixel%d_%s_%d.png'%(self.pixel_size, str(Mode_Index[index]), count)
            print(img_path)
            fig.savefig(img_path)
            plt.title('fig.%d'%count)
            count += 1

        if self.transform:
            sample = self.transform(sample)
            
            if visualize:
                img = sample
                transformed_count = 0
                image_count = 0
                fig = plt.figure()
                fig.set_facecolor('white')
                grid_shift = axes_grid1.AxesGrid(
                fig, 111, nrows_ncols=(1, self.input_channel), axes_pad = 0.5, cbar_location = "right",
                cbar_mode="each", cbar_size="15%", cbar_pad="5%",)  

                speed_array_final = img[0]
                count_final = img[1]
                acc_final = img[2]
                bearing_array_final = img[3]
                bearing_rate_array_final = img[4]
                
                im0 = grid_shift[image_count].imshow(speed_array_final, cmap='jet', interpolation='nearest')
                grid_shift.cbar_axes[0].colorbar(im0)

                im1 = grid_shift[image_count+1].imshow(count_final, cmap='jet', interpolation='nearest')
                grid_shift.cbar_axes[1].colorbar(im1)

                im2 = grid_shift[image_count+2].imshow(acc_final, cmap='jet', interpolation='nearest')
                grid_shift.cbar_axes[2].colorbar(im2)

                im3 = grid_shift[image_count+3].imshow(bearing_array_final, cmap='jet', interpolation='nearest')
                grid_shift.cbar_axes[3].colorbar(im3)

                im4 = grid_shift[image_count+4].imshow(bearing_rate_array_final, cmap='jet', interpolation='nearest')
                grid_shift.cbar_axes[4].colorbar(im4)

                img_path = 'image_transformed_withoutcrop_class_fixed_pixel%d_%s_%d.png'%(self.pixel_size, str(Mode_Index[index]), transformed_count)
                print(img_path)
                fig.savefig(img_path)
                plt.title('fig.%d'%transformed_count)
                transformed_count += 1
            
            # import pdb; pdb.set_trace()

        return sample, label, scale

    def __len__(self):
        return self.n_samples


class TestDataset(Dataset):

    def __init__(self, file_path, pixel_size, input_channel, transform=None):

        self.pixel_size = pixel_size
        self.input_channel = input_channel
        self.transform = transform
        
        num_class = 6
        # filename = '/home/xieyuan/Transportation-mode/Traj2Image/datafiles/Geolife/trips_traj2image_trip_shift_%dclass_pixelsize%d_back0_180bearing_unfixed_TITS.pickle'%(num_class, self.pixel_size)
        with open(file_path, 'rb') as f: 
            kfold_dataset = pickle.load(f) 
        dataset = kfold_dataset
        
        self.test_x_geolife = np.array(dataset[9 + input_channel])
        self.test_y_geolife = np.array(dataset[16])
        self.test_scale_geolife = np.array(dataset[17])
        self.n_samples = self.test_x_geolife.shape[0]

    def __getitem__(self, index):
        sample = self.test_x_geolife[index]
        label = self.test_y_geolife[index]
        scale = self.test_scale_geolife[index]

        if self.transform:
            sample = self.transform(sample)

        return sample, label, scale

    def __len__(self):
        return self.n_samples

class ToTensor:
    def __call__(self, input):
        """
            input : img (5, W, H)
            return: (1, 4, W, H)
        """
        return torch.tensor(input).float()