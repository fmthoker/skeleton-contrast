# sys
import pickle

# torch
import torch
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
np.set_printoptions(threshold=np.inf)
import random

try:
    from feeder import augmentations
except:
    import augmentations


class Feeder(torch.utils.data.Dataset):
    """ 
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
    """

    def __init__(self,
                 data_path,
                 num_frame_path,
                 l_ratio,
                 input_size,
                 input_representation,
                 mmap=True):

        self.data_path = data_path
        self.num_frame_path= num_frame_path
        self.input_size=input_size
        self.input_representation=input_representation
        self.crop_resize =True
        self.l_ratio = l_ratio


        self.load_data(mmap)
        self.N, self.C, self.T, self.V, self.M = self.data.shape
        print(self.data.shape,len(self.number_of_frames))
        print("l_ratio",self.l_ratio)

    def load_data(self, mmap):
        # data: N C V T M

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        # load num of valid frame length
        self.number_of_frames= np.load(self.num_frame_path)

    def __len__(self):
        return self.N

    def __iter__(self):
        return self

    def __getitem__(self, index):

        # get raw input

        # input: C, T, V, M
        data_numpy = np.array(self.data[index])
        number_of_frames = self.number_of_frames[index]

        # apply spatio-temporal augmentations to generate  view 1 

        # temporal crop-resize
        data_numpy_v1_crop = augmentations.temporal_cropresize(data_numpy, number_of_frames, self.l_ratio, self.input_size)


        # randomly select  one of the spatial augmentations 
        flip_prob  = random.random()
        if flip_prob < 0.5:
                 data_numpy_v1 = augmentations.joint_courruption(data_numpy_v1_crop)
        else:
                 data_numpy_v1 = augmentations.pose_augmentation(data_numpy_v1_crop)


        # apply spatio-temporal augmentations to generate  view 2

        # temporal crop-resize
        data_numpy_v2_crop = augmentations.temporal_cropresize(data_numpy,number_of_frames, self.l_ratio, self.input_size)

        # randomly select  one of the spatial augmentations 
        flip_prob  = random.random()
        if flip_prob < 0.5:
                 data_numpy_v2 = augmentations.joint_courruption(data_numpy_v2_crop)
        else:
                 data_numpy_v2 = augmentations.pose_augmentation(data_numpy_v2_crop)


        # convert augmented views into input formats based on skeleton-representations
        if self.input_representation == "seq-based": 

             #Input for sequence-based representation
             # two person  input ---> shpae (64 X 150)

             #View 1
             input_v1 = data_numpy_v1.transpose(1,2,0,3)
             input_v1 = input_v1.reshape(-1,150).astype('float32')

             #View 2
             input_v2 = data_numpy_v2.transpose(1,2,0,3)
             input_v2 = input_v2.reshape(-1,150).astype('float32')

             return input_v1, input_v2

        elif self.input_representation == "graph-based" or self.input_representation == "image-based": 

             #input for graph-based or image-based representation
             # two person input --->  shape (3, 64, 25, 2)

             #View 1
             input_v1 = data_numpy_v1.astype('float32')
             #View 2
             input_v2 = data_numpy_v2.astype('float32')

             return input_v1, input_v2
