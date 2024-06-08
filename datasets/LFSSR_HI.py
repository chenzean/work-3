import os
from os import listdir
from os.path import isfile
import torch
import numpy as np
import torchvision
import torch.utils.data
import PIL
import re
import random
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import h5py
from torch.utils.data import DataLoader


class LFSSR_HI:
    def __init__(self, args):
        self.args = args
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def get_loaders(self, parse_patches=False, validation='snow'):

        print("=> Load light field heterogeneous imaging dataset...")
        train_dataset = TrainSetDataLoader(self.args)
        test_dataset = TestSetDataLoader(self.args)

        train_loader = DataLoader(dataset=train_dataset, num_workers=self.args.num_workers,
                                  batch_size=self.args.training_batch_size, shuffle=True)
        Test_Loaders = DataLoader(dataset=test_dataset, num_workers=self.args.num_workers,
                                  batch_size=self.args.sampling_batch_size, shuffle=False)

        return train_loader, Test_Loaders



class TrainSetDataLoader(Dataset):
    def __init__(self, args):
        super(TrainSetDataLoader, self).__init__()
        self.angRes = args.angRes
        self.scale_factor = args.SR_factor
        self.dataset_dir = args.path_for_train + 'SR_' + str(self.angRes) + 'x' + str(self.angRes) + '_' + \
                           str(self.scale_factor) + 'x/'
        self.ref_dataset_dir = args.path_for_train + 'ref_' + str(self.scale_factor) + 'x/'

        self.data_list = os.listdir(self.dataset_dir)
        self.ref_list = os.listdir(self.ref_dataset_dir)

        self.file_list = []
        self.ref_file_list = []

        # input lr image
        for index, _ in enumerate(self.data_list):
            self.file_list.extend([self.data_list[index]])

        # #  stack of 2D HR ref images
        for index, _ in enumerate(self.ref_list):
            self.ref_file_list.extend([self.ref_list[index]])

        # Determine the length of the dataset
        if len(self.file_list):
            self.item_num = len(self.file_list)
        else:
            print('There are problems with the dataset. (Different number of reference and low-resolution images)')

    def __getitem__(self, index):
        file_name = [self.dataset_dir + self.file_list[index]]
        ref_file_name = [self.ref_dataset_dir + self.ref_file_list[index]]
        Lr_angRes_in = self.angRes
        Lr_angRes_out = self.angRes

        # input LR LF images and 2D HR reference images
        with h5py.File(file_name[0], 'r') as hf1, h5py.File(ref_file_name[0], 'r') as hf2:
            Lr_SAI_y = np.array(hf1.get('Lr_SAI_y')).astype(np.float32) # Lr_SAI_y
            Hr_SAI_y = np.array(hf1.get('Hr_SAI_y')).astype(np.float32) # Hr_SAI_y
            ref_SAI_y = np.array(hf2.get('ref_sai')).astype(np.float32)  # ref_sai_y
            # random_number = random.randint(0, 55)
            # print('随机选取的ref 图像位置', random_number)
            # # print(ref_SAI_y.shape)        # (128, 128, 56)
            # ref_y = ref_SAI_y[:,:,random_number]


            Lr_SAI_y, Hr_SAI_y, ref_y = augmentation(Lr_SAI_y, Hr_SAI_y, ref_SAI_y)
            Lr_SAI_y_out = ToTensor()(Lr_SAI_y.copy())
            Hr_SAI_y_out = ToTensor()(Hr_SAI_y.copy())
            ref_SAI_y_out = ToTensor()(ref_y.copy())


        return Lr_SAI_y_out, Hr_SAI_y_out, ref_SAI_y_out, [Lr_angRes_in, Lr_angRes_out]

    def __len__(self):
        return self.item_num

def augmentation(data, label, ref):
    if random.random() < 0.5:  # flip along W-V direction
        data = data[:, ::-1]
        label = label[:, ::-1]
        ref = ref[:, ::-1]
    if random.random() < 0.5:  # flip along W-V direction
        data = data[::-1, :]
        label = label[::-1, :]
        ref = ref[::-1, :]
    if random.random() < 0.5:  # transpose between U-V and H-W
        data = data.transpose(1, 0)
        label = label.transpose(1, 0)
        ref = ref.transpose(1, 0)
    return data, label, ref

class TestSetDataLoader(Dataset):
    def __init__(self, args):
        super(TestSetDataLoader, self).__init__()
        self.angRes = args.angRes
        self.scale_factor = args.SR_factor
        self.dataset_dir = args.path_for_test + 'SR_' + str(self.angRes) + 'x' + str(self.angRes) + '_' + \
                           str(self.scale_factor) + 'x/'
        self.ref_dataset_dir = args.path_for_test + 'ref_' + str(self.scale_factor) + 'x/'

        self.dataset_list = os.listdir(self.dataset_dir)
        self.ref_list = os.listdir(self.ref_dataset_dir)

        self.file_list = []
        self.ref_file_list = []

        # self.random_number = random.randint(0, 55)
        # The 'random_number'st in the 2D HR image stack is taken out as the 2D reference image
        # print(f"The '{self.random_number}'st in the 2D HR image stack is taken out as the 2D reference image")

        # input LR LF images
        for index, _ in enumerate(self.dataset_list):
            self.file_list.extend([self.dataset_list[index]])

        #  stack of 2D HR ref images
        for index, _ in enumerate(self.ref_list):
            self.ref_file_list.extend([self.ref_list[index]])

        # Determine the length of the dataset
        if len(self.file_list)==len(self.ref_file_list):
            self.item_num = len(self.file_list)
        else:
            print('There are problems with the dataset. (Different number of reference and low-resolution images)')

    def __getitem__(self, index):
        file_name = [self.dataset_dir + self.file_list[index]]
        ref_file_name = [self.ref_dataset_dir + self.ref_file_list[index]]

        with h5py.File(file_name[0], 'r') as hf1, h5py.File(ref_file_name[0], 'r') as hf2:

            Lr_SAI_y = np.array(hf1.get('Lr_SAI_y')).astype(np.float32)
            Hr_SAI_y = np.array(hf1.get('Hr_SAI_y')).astype(np.float32)
            Sr_SAI_cbcr = np.array(hf1.get('Sr_SAI_cbcr'), dtype='single')
            ref_SAI_y = np.array(hf2.get('ref_y')).astype(np.float32)  # ref_sai_y
            # print(ref_SAI_y.shap e)
            # ref_y = ref_SAI_y[:, :, self.random_number]
            ref_y = ref_SAI_y

            Lr_SAI_y = np.transpose(Lr_SAI_y, (1, 0))
            Hr_SAI_y = np.transpose(Hr_SAI_y, (1, 0))
            Sr_SAI_cbcr  = np.transpose(Sr_SAI_cbcr,  (2, 1, 0))
            ref_y = np.transpose(ref_y, (1, 0))

        Lr_SAI_y = ToTensor()(Lr_SAI_y.copy())
        Hr_SAI_y = ToTensor()(Hr_SAI_y.copy())
        Sr_SAI_cbcr = ToTensor()(Sr_SAI_cbcr.copy())
        ref_y = ToTensor()(ref_y.copy())

        Lr_angRes_in = self.angRes
        Lr_angRes_out = self.angRes
        LF_name = self.file_list[index].split('/')[-1].split('.')[0]

        return Lr_SAI_y, Hr_SAI_y, ref_y, Sr_SAI_cbcr, [Lr_angRes_in, Lr_angRes_out], LF_name

    def __len__(self):
        return self.item_num
