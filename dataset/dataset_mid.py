import os
import os.path as osp
import numpy as np
from PIL import Image

import torch
import torchvision
from torch.utils import data

import glob

class DAVIS_MO_Test(data.Dataset):
    # for multi object, do shuffling

    def __init__(self, root, imset='2017/train.txt', resolution='480p', single_object=False,direction_m=False):
        self.root = root
        self.mask_dir = '/jmain02/home/J2AD001/wwp01/axd53-wwp01/codes/results'
        self.mask480_dir = '/jmain02/home/J2AD001/wwp01/axd53-wwp01/codes/results'
        self.image_dir = os.path.join(root, 'JPEGImages', resolution)
        _imset_dir = os.path.join(root, 'ImageSets')
        _imset_f = os.path.join(_imset_dir, imset)
        self.direction_m = direction_m
        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.shape = {}
        self.size_480p = {}
        self.start_frame_for_tracking_len = {}
        with open(os.path.join(_imset_f), "r") as lines:
            for line in lines:
                _video = line.rstrip('\n')
                self.videos.append(_video)
                self.start_frame_for_tracking_len[_video] = int(len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg')))/2)
                start_frame_for_tracking = sorted(glob.glob(os.path.join(self.image_dir,_video,"*.jpg")),reverse=self.direction_m)[self.start_frame_for_tracking_len[_video]]
                #print ('video:',_video)
                #print ('Length:',len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg'))))
                #print ('selected start frame index:',self.start_frame_for_tracking_len[_video])
                #print ('start frame tracking:',start_frame_for_tracking)
                
                self.num_frames[_video] = len(glob.glob(os.path.join(self.image_dir, _video, '*.jpg'))) - self.start_frame_for_tracking_len[_video]
                #print(self.num_frames[_video])
                _img = np.array(Image.open(glob.glob(os.path.join(self.image_dir,_video,"*.jpg"))[0]))
                self.shape[_video] = np.shape(_img)[0:2]


        self.K = 15
        self.single_object = single_object

    def __len__(self):
        return len(self.videos)


    def To_onehot(self, mask):
        M = np.zeros((self.K, mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for k in range(self.K):
            M[k] = (mask == k).astype(np.uint8)
        return M
    
    def All_to_onehot(self, masks):
        Ms = np.zeros((self.K, masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
        for n in range(masks.shape[0]):
            Ms[:,n] = self.To_onehot(masks[n])
        return Ms

    def __getitem__(self, index):
        
        if not isinstance(index, int):
            x,y = index
            video = self.videos[x]
            info = {}
            info['name'] = video
            return info,info #two times just to keep number of output to 2
        else:
            video = self.videos[index]
            _mask = np.array(Image.open(sorted(glob.glob(os.path.join(self.mask_dir, video, '*.png')),reverse=self.direction_m)[self.start_frame_for_tracking_len[video]]).convert("P"))
            #print(sorted(glob.glob(os.path.join(self.mask_dir, _video, '*.png')))[0])
            self.num_objects[video] = np.max(_mask)
            #print ('MAX:::', self.num_objects[video])

            
            info = {}
            info['name'] = video
            info['num_frames'] = self.num_frames[video]
            #info['size_480p'] = self.size_480p[video]
            info['start_frame'] = int(sorted(glob.glob(os.path.join(self.image_dir,video,"*.jpg")),reverse=self.direction_m)[self.start_frame_for_tracking_len[video]].split("/")[-1][-14:-4])


            N_frames = np.empty((self.num_frames[video],)+self.shape[video]+(3,), dtype=np.float32)
            N_masks = np.empty((self.num_frames[video],)+self.shape[video], dtype=np.uint8)


            num_objects = torch.LongTensor([int(self.num_objects[video])])
            return num_objects, info

    def load_single_image(self,video,f):
        import glob
        #print("F=:",f)
        #print("DIR",self.image_dir)
        #print("self len: ",self.start_frame_for_tracking_len)
        #print(sorted(glob.glob(os.path.join(self.image_dir, video,"*.jpg")))[f])
        N_frames = np.empty((1,)+self.shape[video]+(3,), dtype=np.float32)
        N_masks = np.empty((1,)+self.shape[video], dtype=np.uint8)

        img_file = sorted(glob.glob(os.path.join(self.image_dir, video,"*.jpg")),reverse=self.direction_m)[f+self.start_frame_for_tracking_len[video]]
        #print("Loaded image >>>>>>",img_file)

        N_frames[0] = np.array(Image.open(img_file).convert('RGB'))/255.
        try:
            mask_file = sorted(glob.glob(os.path.join(self.mask_dir, video,"*.png")),reverse=self.direction_m)[f+self.start_frame_for_tracking_len[video]] # -1 since last and fist frames are not part of the png data
            #print("MASK >>>>>>>>>",mask_file)
            N_masks[0] = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
            
        except:
            N_masks[0] = 255
        Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()
        
        if self.single_object:
            N_masks = (N_masks > 0.5).astype(np.uint8) * (N_masks < 255).astype(np.uint8)
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            num_objects = torch.LongTensor([int(1)])
            return Fs, Ms, num_objects
        else:
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            num_objects = torch.LongTensor([int(self.num_objects[video])])
            return Fs, Ms      

    def load_single_image_reserse(self,video,f):
        import glob
        #print("F=:",f)
        #print("DIR",self.image_dir)
        #print(sorted(glob.glob(os.path.join(self.image_dir, video,"*.jpg")))[f])
        N_frames = np.empty((1,)+self.shape[video]+(3,), dtype=np.float32)
        N_masks = np.empty((1,)+self.shape[video], dtype=np.uint8)

        img_file = sorted(glob.glob(os.path.join(self.image_dir, video,"*.jpg")),reverse= not self.direction_m)[f+self.start_frame_for_tracking_len[video]]
        #print(f"Added dded image of video {video} is {img_file}")

        N_frames[0] = np.array(Image.open(img_file).convert('RGB'))/255.
        try:
            mask_file = sorted(glob.glob(os.path.join(self.mask_dir, video,"*.png")),reverse= not self.direction_m)[f+self.start_frame_for_tracking_len[video]]
            #print("Added MASK",mask_file)
            N_masks[0] = np.array(Image.open(mask_file).convert('P'), dtype=np.uint8)
        except:
            N_masks[0] = 255
        Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()
        
        if self.single_object:
            N_masks = (N_masks > 0.5).astype(np.uint8) * (N_masks < 255).astype(np.uint8)
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            num_objects = torch.LongTensor([int(1)])
            return Fs, Ms, num_objects
        else:
            Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
            num_objects = torch.LongTensor([int(self.num_objects[video])])
            return Fs, Ms      


    def load_single_image_name(self,video,f):
        import glob
        file_name_jpg = sorted(glob.glob(os.path.join(self.image_dir, video,"*.jpg")),reverse=self.direction_m)[f+self.start_frame_for_tracking_len[video]]
        return file_name_jpg.split("/")[-1][:-4]

if __name__ == '__main__':
    pass
