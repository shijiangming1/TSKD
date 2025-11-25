import numpy as np
from PIL import Image
import torch.utils.data as data
from ChannelAug import ChannelAdap, ChannelAdapGray, ChannelRandomErasing
import torchvision.transforms as transforms
import random
import os

random.seed(1)


class ChannelExchange(object):
    """ Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, gray=2):
        self.gray = gray

    def __call__(self, img):

        idx = random.randint(0, self.gray)

        if idx == 0:
            # random select R Channel
            img[1, :, :] = img[0, :, :]
            img[2, :, :] = img[0, :, :]
        elif idx == 1:
            # random select B Channel
            img[0, :, :] = img[1, :, :]
            img[2, :, :] = img[1, :, :]
        elif idx == 2:
            # random select G Channel
            img[0, :, :] = img[2, :, :]
            img[1, :, :] = img[2, :, :]
        else:
            tmp_img = 0.2989 * img[0, :, :] + 0.5870 * img[1, :, :] + 0.1140 * img[2, :, :]
            img[0, :, :] = tmp_img
            img[1, :, :] = tmp_img
            img[2, :, :] = tmp_img
        return img


class SYSUData(data.Dataset):
    def __init__(self,data_dir, transform=None, colorIndex=None, thermalIndex=None, noise_rate=0., noise_file='', mode='', probV=[], probI=[]):

        imgw, imgh=192,384
        data_dir = data_dir
        # Load training images (path) and labels
        # train_color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        # self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')
        # # self.train_color_label = np.load('/data1/SSM/SMMD_supp/save_model/precess/rgb_P_whole_label_rate_0.50.npy')
        #
        # train_thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')
        # self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')

        train_color_image = np.load('/data/sjm/SYSU-MM01train_rgb_resized_img.npy')
        self.train_color_label = np.load('/data/sjm/SYSU-MM01train_rgb_resized_label.npy')
        # self.train_color_label = np.load('/data1/SSM/SMMD_supp/save_model/precess/rgb_P_whole_label_rate_0.50.npy')

        train_thermal_image = np.load('/data/sjm/SYSU-MM01train_ir_resized_img.npy')
        self.train_thermal_label = np.load('/data/sjm/SYSU-MM01train_ir_resized_label.npy')
        # self.train_thermal_label = np.load('/data1/SSM/SMMD_supp/save_model/precess/ir_P_whole_label_rate_0.50.npy')

        # #label dataset
        # train_color_image_l = np.load(data_dir + 'train_rgb_resized_img_rate_0.5.npy')
        # train_color_label_l = np.load(data_dir + 'train_rgb_resized_label_rate_0.5.npy')
        # train_thermal_image_l= np.load(data_dir + 'train_ir_resized_img_rate_0.5.npy')
        # train_thermal_label_l = np.load(data_dir + 'train_ir_resized_label_rate_0.5.npy')
        #
        # #unlabel
        # train_color_image_u = np.load(data_dir + 'un_train_rgb_resized_img_rate_0.5.npy')
        # train_color_label_u = np.load(data_dir + 'un_train_rgb_resized_pesudolabel_rate_0.5.npy')
        #
        # train_thermal_image_u = np.load(data_dir + 'un_train_ir_resized_img_rate_0.5.npy')
        # train_thermal_label_u = np.load(data_dir + 'un_train_ir_resized_pesudolabel_rate_0.5.npy')
        #
        # train_color_image=np.concatenate((train_color_image_l, train_color_image_u), axis=0)
        # self.train_color_label=np.concatenate((train_color_label_l, train_color_label_u), axis=0).astype(int)
        # train_thermal_image=np.concatenate((train_thermal_image_l, train_thermal_image_u), axis=0)
        # self.train_thermal_label=np.concatenate((train_thermal_label_l, train_thermal_label_u), axis=0).astype(int)
        self.mode = mode
        self.probI = probI
        self.probV = probV

        print("train with %.2f noisy rates" % noise_rate)

        if noise_rate == 0.:
            self.rgb_cleanIdx = range(len(self.train_color_label))
            self.rgb_noiseIdx = []
            self.ir_cleanIdx = range(len(self.train_thermal_label))
            self.ir_noiseIdx = []
            self.true_train_color_label = self.train_color_label
            self.true_train_thermal_label = self.train_thermal_label
        else:
            if os.path.exists((noise_file + '_rgb.npy')):
                print("loading files and idx of noisy labels")
                self.train_color_label = np.load((noise_file + '_rgb.npy'))
                self.train_thermal_label = np.load((noise_file + '_ir.npy'))
                self.rgb_noiseIdx = np.load((noise_file + '_rgb_noiseIdx.npy'))
                self.ir_noiseIdx = np.load((noise_file + '_ir_noiseIdx.npy'))
                self.rgb_cleanIdx = np.load((noise_file + '_rgb_cleanIdx.npy'))
                self.ir_cleanIdx = np.load((noise_file + '_ir_cleanIdx.npy'))
                self.true_train_color_label = np.load((noise_file + '_rgb_true.npy'))
                self.true_train_thermal_label = np.load((noise_file + '_ir_true.npy'))

            else:  # inject noise
                for j in [0, 1]:
                    noise_idx=[]
                    if j == 0:
                        ids = self.train_color_label[:]
                        self.true_train_color_label = ids.copy()
                    else:
                        ids = self.train_thermal_label[:]
                        self.true_train_thermal_label = ids.copy()
                    tmp_list = ids.copy()
                    unique_id = np.unique(ids)
                    for id_single in unique_id:
                        # arr_ids = np.array(ids)
                        single_index = np.where(ids == id_single)
                        single_index=list(single_index[0])
                        random.shuffle(single_index)
                        # noise_idx_p=random.sample(range(len(single_index)), int(np.ceil(noise_rate * len(single_index))))
                        noise_idx_p=single_index[0:int(np.ceil(noise_rate * len(single_index)))]
                        noise_idx.extend(noise_idx_p)
                    noise_idx=np.array(noise_idx)
                    noise_idx.sort()
                    clean_idx = list(set(range(len(ids))).difference(set(noise_idx)))

                    random.seed()
                    for i in noise_idx:
                        tmp = random.choice(unique_id)
                        while ids[i] == tmp:
                            tmp = random.choice(unique_id)
                        ids[i] = tmp

                    if j == 0:
                        self.train_color_label = ids.copy()
                        self.rgb_noiseIdx = np.array(noise_idx)
                        self.rgb_cleanIdx = np.array(clean_idx)
                        np.save((noise_file + '_rgb_cleanIdx.npy'), self.rgb_cleanIdx)
                        np.save((noise_file + '_rgb_noiseIdx.npy'), self.rgb_noiseIdx)
                        print("save rgb noisy labels to %s ..." % (noise_file + '_rgb.npy'))
                        np.save((noise_file + '_rgb.npy'), self.train_color_label)
                        np.save((noise_file + '_rgb_true.npy'), self.true_train_color_label)  # save the real label
                    else:
                        self.train_thermal_label = ids.copy()
                        self.ir_noiseIdx = np.array(noise_idx)
                        self.ir_cleanIdx = np.array(clean_idx)
                        np.save((noise_file + '_ir_cleanIdx.npy'), self.ir_cleanIdx)
                        np.save((noise_file + '_ir_noiseIdx.npy'), self.ir_noiseIdx)
                        print("save ir noisy labels to %s ..." % (noise_file + '_ir.npy'))
                        np.save((noise_file + '_ir.npy'), self.train_thermal_label)
                        np.save((noise_file + '_ir_true.npy'), self.true_train_thermal_label)
        # BGR to RGB
        if mode == 'clean':
            self.train_color_image = train_color_image[self.rgb_cleanIdx]
            self.train_thermal_image = train_thermal_image[self.ir_cleanIdx]
            self.train_color_label = self.train_color_label[self.rgb_cleanIdx]
            self.train_thermal_label = self.train_thermal_label[self.ir_cleanIdx]
            self.true_train_color_label = self.train_color_label[self.rgb_cleanIdx]
            self.true_train_thermal_label = self.train_thermal_label[self.ir_cleanIdx]
        elif mode == 'pre_train':
            self.train_color_image = train_color_image[self.rgb_cleanIdx,:,:,:]
            self.train_thermal_image = train_thermal_image[self.ir_cleanIdx,:,:,:]
            self.true_train_color_label = self.train_color_label[self.rgb_cleanIdx]
            self.true_train_thermal_label = self.train_thermal_label[self.ir_cleanIdx]
            self.train_color_label = self.train_color_label[self.rgb_cleanIdx]
            self.train_thermal_label = self.train_thermal_label[self.ir_cleanIdx]
        elif mode == 'create_pesudo':
            self.train_color_image = train_color_image[self.rgb_noiseIdx,:,:,:]
            self.train_thermal_image = train_thermal_image[self.ir_noiseIdx,:,:,:]
            # self.true_train_color_label = self.train_color_label[self.rgb_cleanIdx]
            # self.true_train_thermal_label = self.train_thermal_label[self.ir_cleanIdx]
            self.true_train_color_label = self.true_train_color_label[self.rgb_noiseIdx]
            self.true_train_thermal_label = self.true_train_thermal_label[self.ir_noiseIdx]
            self.train_color_label = self.train_color_label[self.rgb_noiseIdx]
            self.train_thermal_label = self.train_thermal_label[self.ir_noiseIdx]
        else:
            self.train_color_image = train_color_image
            self.train_thermal_image = train_thermal_image

        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform_thermal = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((imgh, imgw)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability=0.5),
            ChannelAdapGray(probability=0.5)])

        self.transform_color = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((imgh, imgw)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomGrayscale(p = 0.1),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability=0.5)])

        self.transform_color1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((imgh, imgw)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability=0.5),
            ChannelExchange(gray=2)])
        self.transform_create_pesudelabels = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((imgh, imgw)),
            transforms.ToTensor(),
            normalize,
        ])
    def __getitem__(self, index):

        img1, target1, true_target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[
            self.cIndex[index]], self.true_train_color_label[self.cIndex[index]]
        img2, target2, true_target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[
            self.tIndex[index]], self.true_train_thermal_label[self.tIndex[index]]
        ori_img1=self.transform_create_pesudelabels(img1)
        ori_img2=self.transform_create_pesudelabels(img2)

        img1_0 = self.transform_color(img1)
        img1_1 = self.transform_color1(img1)
        img2 = self.transform_thermal(img2)

        if self.mode == 'pre_train':
            return img1_0, img1_1, img2, target1, target2
        elif self.mode == 'create_pesudo':
            return ori_img1, ori_img2, true_target1, true_target2, self.rgb_noiseIdx[self.cIndex[index]], self.ir_noiseIdx[self.tIndex[index]]
        if self.mode == 'warmup':
            return img1_0, img1_1, img2, target1, target2
        elif self.mode == 'evaltrain':
            return img1_0, img1_1, img2, target1, target2, self.cIndex[index], self.tIndex[index]
        elif self.mode == 'train':
            probV_1, probV_2, probI = self.probV_1[self.cIndex[index]], self.probV_2[self.cIndex[index]], self.probI[self.tIndex[index]]
            return img1_0, img1_1, img2, target1, target2, true_target1, true_target2, probV_1, probV_2, probI

        elif self.mode == 'evalpair':
            return img1_0, img1_1, img2, target1, target2, true_target1, true_target2


class RegDBData(data.Dataset):
    def __init__(self, data_dir, trial, transform=None, colorIndex=None, thermalIndex=None, noise_rate=0.,
                 noise_file='', mode='', probV_1=[], probV_2=[], probI=[]):
        # Load training images (path) and labels
        data_dir = data_dir
        train_color_list = data_dir + 'idx/train_visible_{}'.format(trial) + '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial) + '.txt'
        color_img_file, self.train_color_label = load_data(train_color_list)
        thermal_img_file, self.train_thermal_label = load_data(train_thermal_list)

        train_color_image = []
        for i in range(len(color_img_file)):
            img = Image.open(data_dir + color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image)

        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir + thermal_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)

        self.mode = mode
        self.probI = probI
        self.probV_1 = probV_1
        self.probV_2 = probV_2

        print("train with %.2f noisy rates" % noise_rate)

        if noise_rate == 0.:
            print("loading files and idx of trial {}".format(trial))
            self.rgb_cleanIdx = range(len(self.train_color_label))
            self.rgb_noiseIdx = []
            self.ir_cleanIdx = range(len(self.train_thermal_label))
            self.ir_noiseIdx = []
            self.true_train_color_label = self.train_color_label
            self.true_train_thermal_label = self.train_thermal_label
        else:
            if os.path.exists((noise_file +'_trial{}_'.format(trial) + 'rgb.npy')):
                print("loading files and idx of noisy labels of trial {}".format(trial))
                self.train_color_label = np.load((noise_file + '_trial{}_'.format(trial) + 'rgb.npy'))
                self.train_thermal_label = np.load((noise_file + '_trial{}_'.format(trial) + 'ir.npy'))
                self.rgb_noiseIdx = np.load((noise_file + '_trial{}_'.format(trial) + 'rgb_noiseIdx.npy'))
                self.ir_noiseIdx = np.load((noise_file + '_trial{}_'.format(trial) + 'ir_noiseIdx.npy'))
                self.rgb_cleanIdx = np.load((noise_file + '_trial{}_'.format(trial) + 'rgb_cleanIdx.npy'))
                self.ir_cleanIdx = np.load((noise_file + '_trial{}_'.format(trial) + 'ir_cleanIdx.npy'))
                self.true_train_color_label = np.load((noise_file + '_trial{}_'.format(trial) + 'rgb_true.npy'))
                self.true_train_thermal_label = np.load((noise_file + '_trial{}_'.format(trial) + 'ir_true.npy'))

            else:  # inject noise
                num_class = 0
                while num_class != np.unique(self.train_color_label).size:
                    for j in [0, 1]:
                        if j == 0:
                            ids = self.train_color_label[:]
                            self.true_train_color_label = ids.copy()
                        else:
                            ids = self.train_thermal_label[:]
                            self.true_train_thermal_label = ids.copy()
                        tmp_list = ids.copy()
                        unique_id = np.unique(ids)
                        noise_idx = (random.sample(range(len(ids)), int(np.ceil(noise_rate * len(ids)))))
                        noise_idx.sort()
                        clean_idx = list(set(range(len(ids))).difference(set(noise_idx)))

                        random.seed()
                        for i in noise_idx:
                            tmp = random.choice(unique_id)
                            while ids[i] == tmp:
                                tmp = random.choice(unique_id)
                            ids[i] = tmp

                        num_class = np.unique(ids).size
                        if num_class != np.unique(self.train_color_label).size:
                            break

                        if j == 0:
                            self.train_color_label = ids.copy()
                            self.rgb_noiseIdx = np.array(noise_idx)
                            self.rgb_cleanIdx = np.array(clean_idx)
                            np.save((noise_file + '_trial{}_'.format(trial) + 'rgb_cleanIdx.npy'), self.rgb_cleanIdx)
                            np.save((noise_file + '_trial{}_'.format(trial) + 'rgb_noiseIdx.npy'), self.rgb_noiseIdx)
                            print("save rgb noisy labels to %s ..." % (noise_file + '_rgb.npy'))
                            np.save((noise_file + '_trial{}_'.format(trial) + 'rgb.npy'), self.train_color_label)
                            np.save((noise_file + '_trial{}_'.format(trial) + 'rgb_true.npy'), self.true_train_color_label)  # save the real label
                        else:
                            self.train_thermal_label = ids.copy()
                            self.ir_noiseIdx = np.array(noise_idx)
                            self.ir_cleanIdx = np.array(clean_idx)
                            np.save((noise_file + '_trial{}_'.format(trial) + 'ir_cleanIdx.npy'), self.ir_cleanIdx)
                            np.save((noise_file + '_trial{}_'.format(trial) + 'ir_noiseIdx.npy'), self.ir_noiseIdx)
                            print("save ir noisy labels to %s ..." % (noise_file + '_ir.npy'))
                            np.save((noise_file + '_trial{}_'.format(trial) + 'ir.npy'), self.train_thermal_label)
                            np.save((noise_file + '_trial{}_'.format(trial) + 'ir_true.npy'), self.true_train_thermal_label)
        # BGR to RGB
        if mode == 'clean':
            self.train_color_image = train_color_image[self.rgb_cleanIdx]
            self.train_thermal_image = train_thermal_image[self.ir_cleanIdx]
            self.train_color_label = self.train_color_label[self.rgb_cleanIdx]
            self.train_thermal_label = self.train_thermal_label[self.ir_cleanIdx]
            self.true_train_color_label = self.train_color_label[self.rgb_cleanIdx]
            self.true_train_thermal_label = self.train_thermal_label[self.ir_cleanIdx]
        elif mode == 'pretrain':
            self.train_color_image = train_color_image[self.rgb_cleanIdx, :, :, :]
            self.train_thermal_image = train_thermal_image[self.ir_cleanIdx, :, :, :]
            self.true_train_color_label = self.train_color_label[self.rgb_cleanIdx]
            self.true_train_thermal_label = self.train_thermal_label[self.ir_cleanIdx]
            self.train_color_label = self.train_color_label[self.rgb_cleanIdx]
            self.train_thermal_label = self.train_thermal_label[self.ir_cleanIdx]
        elif mode == 'create_pesudo':
            self.train_color_image = train_color_image[self.rgb_noiseIdx, :, :, :]
            self.train_thermal_image = train_thermal_image[self.ir_noiseIdx, :, :, :]
            # self.true_train_color_label = self.train_color_label[self.rgb_cleanIdx]
            # self.true_train_thermal_label = self.train_thermal_label[self.ir_cleanIdx]
            self.true_train_color_label = self.true_train_color_label[self.rgb_noiseIdx]
            self.true_train_thermal_label = self.true_train_thermal_label[self.ir_noiseIdx]
            self.train_color_label = self.train_color_label[self.rgb_noiseIdx]
            self.train_thermal_label = self.train_thermal_label[self.ir_noiseIdx]
        else:
            self.train_color_image = train_color_image
            self.train_thermal_image = train_thermal_image

        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform_thermal = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability=0.5),
            ChannelAdapGray(probability=0.5)])

        # self.transform_thermal2 = transforms.Compose([
        #         transforms.ToPILImage(),
        #         transforms.ColorJitter(brightness=0.5),
        #         transforms.Pad(10),
        #         transforms.RandomCrop((H, W)),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         normalize,
        #         ChannelRandomErasing(probability=0.5),
        #         ChannelT(probability=0.5)
        #     ])

        self.transform_color = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomGrayscale(p = 0.1),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability=0.5)])

        self.transform_color1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((288, 144)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability=0.5),
            ChannelExchange(gray=2)])
        self.transform_create_pesudelabels = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((288, 144)),
            transforms.ToTensor(),
            normalize,
        ])

    def __getitem__(self, index):

        img1, target1, true_target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[
            self.cIndex[index]], self.true_train_color_label[self.cIndex[index]]
        img2, target2, true_target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[
            self.tIndex[index]], self.true_train_thermal_label[self.tIndex[index]]
        ori_img1 = self.transform_create_pesudelabels(img1)
        ori_img2 = self.transform_create_pesudelabels(img2)
        img1_0 = self.transform_color(img1)
        img1_1 = self.transform_color1(img1)
        img2 = self.transform_thermal(img2)

        if self.mode == 'warmup':
            return img1_0, img1_1, img2, target1, target2
        elif self.mode == 'pretrain':

            return img1_0, img1_1, img2, target1, target2
        elif self.mode == 'create_pesudo':
            return ori_img1, ori_img2, true_target1, true_target2, self.rgb_noiseIdx[self.cIndex[index]], \
                   self.ir_noiseIdx[self.tIndex[index]]
        elif self.mode == 'evaltrain':
            return img1_0, img1_1, img2, target1, target2, self.cIndex[index], self.tIndex[index]

        elif self.mode == 'train':
            probV_1, probV_2, probI = self.probV_1[self.cIndex[index]], self.probV_2[self.cIndex[index]], self.probI[
                self.tIndex[index]]
            return img1_0, img1_1, img2, target1, target2, true_target1, true_target2, probV_1, probV_2, probI


class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size=(144, 288)):
        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1, target1 = self.test_image[index], self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)


def load_data(input_data_path):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]

    return file_image, file_label
