################################################################################
#
#  Main library for neu-net segmentation.  This implements a standard u-net
#  model where images are processed in each slicing orientation.
#
#  Author: Ryan Cabeen
#
################################################################################


import os, sys, json, glob, shutil

import numpy as np
import nibabel as nib
import scipy.io as io
import scipy.ndimage as snd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from torchsummary import summary
import albumentations as album

########################################################################################################################
# Main Entry Points                                                                                                    #
########################################################################################################################

def train_main(settings, init, images, masks, output):
    '''
    Trains a u-net given a pair of corresponding images and segmentation masks.

    Parameters:
        settings (Settings): the settings object for training the model
        init (str): a optional path to an initial model for training (may be None)
        images (str): a string storing the path to the training image directory
        masks (str): a string storing the path to the training mask directory
        output (str): a string storing the path to the expected output directory

    Parameters:
        Nothing
    '''

    print("started training")

    os.makedirs(output, exist_ok=True)
    use_gpu = torch.cuda.is_available()
    my_model = UNet2d(settings)

    if init is not None:
        print("loading initial model")
        checkpoint = torch.load(init, map_location={'cuda:0':'cpu'})
        settings.rescale = checkpoint['rescale']
        settings.kernel = checkpoint['kernel']

        my_model = UNet2d(settings)
        my_model.load_state_dict(checkpoint['state'])
    else:
        print("creating initial model")
        my_model = UNet2d(settings)

    device="cpu"
    if use_gpu:
        my_model.cuda()
        cudnn.benchmark = True
        device = "cuda"

    print("")
    print("Settings")
    print("========")
    print("  channels: %d" % settings.channels)
    print("  labels: %d" % settings.labels)
    print("  augment: %d" % settings.augment)
    print("  rescale: %d" % settings.rescale)
    print("  kernel: %d" % settings.kernel)
    print("  epochs: %d" % settings.epochs)
    print("  rate: %d" % settings.rate)
    print("  largest: %d" % settings.largest)
    print("  raw: %d" % settings.raw)
    print("  batches: %d" % settings.batches)
    print("  device: %s" % device)
    print("")
    print("Model")
    print("=====")
    summary(my_model, (settings.channels, settings.rescale, settings.rescale), device=device)
    
    my_optimizer = optim.Adam(my_model.parameters(), lr=settings.rate)
    my_criterion = nn.CrossEntropyLoss()

    if use_gpu:
        my_criterion.cuda()
    
    if not os.path.exists(output):
        os.mkdir(output)
    
    losses = list()
    
    for epoch in range(0, settings.epochs):
        my_loss_v = []
        print("starting epoch %d" % epoch)

        volume_loader = DataLoader(dataset=VolumeDataset(settings, images=images, masks=masks), \
          batch_size=1, shuffle=True, num_workers=0)
    
        for i, (image, mask) in enumerate(volume_loader):

            slice_loader = DataLoader(dataset=SliceDataset(settings, image=image, mask=mask), \
              batch_size=settings.batches, shuffle=True, num_workers=0)

            for j, (my_image_slice, my_mask_slice) in enumerate(slice_loader):
                my_mask_slice = my_mask_slice[:,0,:,:]
                my_image_slice, my_mask_slice = Variable(my_image_slice), Variable(my_mask_slice)

                if use_gpu:
                    my_image_slice = my_image_slice.cuda()
                    my_mask_slice = my_mask_slice.cuda()

                my_predict_slice = my_model(my_image_slice)
                my_loss = my_criterion(my_predict_slice, my_mask_slice)
                my_optimizer.zero_grad()
                my_loss.backward()
                my_optimizer.step()
    
                if use_gpu:
                    my_loss = my_loss.cpu()
    
                my_loss_v.append(my_loss.data.detach().numpy())
                print('\tEpoch:%.2d [%.3d, %.4d]\tLoss: %.6f' % (epoch, i, j * settings.batches, my_loss.data.detach()))

        loss = np.array(my_loss_v).sum()
        losses.append(loss)
        print("\tEpoch: %d; Loss: %.4f" % (epoch, loss))
    
        checkpoint = dict(vars(settings))
        checkpoint['epoch'] = epoch
        checkpoint['state'] = my_model.state_dict()
        checkpoint['optimizer'] = my_optimizer.state_dict()
        checkpoint['loss'] = my_loss_v
        torch.save(checkpoint, os.path.join(output, 'checkpoint-%.3d-model' % (epoch + 1)))
        print("finished epoch %d" % epoch)

    print("saving loss log")
    with open(os.path.join(output, "losses.csv"), "w") as handle:
        handle.write("epoch,loss\n")
        for i in range(len(losses)):
            handle.write("%d,%g\n" % (i, losses[i]))

    print("finished training")

def validate_main(models, images, masks, output):
    '''
    Validates a u-net given a pair of corresponding images and segmentation
    masks.  Selects the model with the best accuracy and saves it as "best-model".

    Parameters:
        models (str): a path to directory of model checkpoints
        images (str): a string storing the path to the validation image directory
        masks (str): a string storing the path to the validation mask directory
        output (str): a string storing the path to the expected output directory

    Parameters:
        Nothing
    '''

    print("started validation")
    os.makedirs(output, exist_ok=True)

    handle = open(os.path.join(output, "dice.csv"), "w")
    handle.write("model,image,dice\n")

    best_dice = 0
    best_model = None

    for model in sorted(glob.glob(os.path.join(models, "*model"))):
        print("validating model %s" % model)
        all_dice = evaluate(model, images, masks, output)
        mean_dice = np.array([v for v in all_dice.values()]).mean()

        handle.write("%s,mean,%g\n" % (model, mean_dice))
        for img in all_dice.keys():
            handle.write("%s,%s,%g\n" % (model, img, all_dice[img]))
        print("mean dice = %g" % mean_dice)
        
        if mean_dice > best_dice:
            print("found new best model")
            best_dice = mean_dice
            best_model = model

    handle.close()

    if best_model:
        shutil.copyfile(best_model, os.path.join(output, "best-model"))
        shutil.copyfile(best_model, os.path.join(output, os.path.basename(best_model)))

        print("best model: %s" % best_model)
        print("best dice: %s" % best_dice)
    else:
        print("no models found")

    print("finished validation")

def test_main(model, images, masks, output):
    '''
    Tests a u-net given a pair of corresponding images and segmentation
    masks.  This may be used to estimate the segmentation accuracy.

    Parameters:
        model (str): a path to a u-net model, e.g. "best-model" from validation
        images (str): a string storing the path to the validation image directory
        masks (str): a string storing the path to the validation mask directory
        output (str): a string storing the path to the expected output directory

    Parameters:
        Nothing
    '''

    print("started testing")
    os.makedirs(output, exist_ok=True)
    dicemap = evaluate(model, images, masks, output)
    dice_array = np.array([v for v in dicemap.values()])

    print("\t%.4f +/- %.4f" % (dice_array.mean(), dice_array.std()))
    with open(os.path.join(output, "dice.csv"), "w") as handle:
        handle.write("image,dice\n")
        for img in dicemap.keys():
            handle.write("%s,%g\n" % (img, dicemap[img]))
    print("finished testing")

def predict_main(model, image, output):
    '''
    Apply a u-net to a given image to produce a predicted segmentation result.

    Parameters:
        model (str): a path to a u-net model, e.g. "best-model" from validation
        image (str): a string storing the path to a nifti image to segment
        output (str): a string storing the path to the expected output nifti mask 

    Parameters:
        Nothing
    '''

    print("started prediction")
    my_settings, my_model = load(model)
    use_gpu = torch.cuda.is_available()
    my_model_on_gpu = next(my_model.parameters()).is_cuda

    if use_gpu:
        if not my_model_on_gpu:
            my_model.cuda()
    else:
        if my_model_on_gpu:
            my_model.cpu()
    
    my_nii = nib.load(image)
    my_image = np.array(my_nii.get_data(), dtype=np.float32)
    if len(my_image) == 3:
        my_image = np.expand_dims(my_image, axis=0)
    else:
        my_image = np.transpose(my_image, (3, 0, 1, 2))

    if not my_settings.raw:
        my_image = np.nan_to_num(my_image, nan=0, posinf=0, neginf=0)
        for i in range(my_settings.channels):
            my_image[i] = (my_image[i] - my_image[i].mean()) / (my_image[i].std() + 1e-6)
        my_image[my_image < -10] = 0
        my_image[my_image >  10] = 0

    my_image = torch.from_numpy(my_image)
    my_image = torch.unsqueeze(my_image, 0)

    my_predict_mask = predict(my_settings, my_model, my_image) > 0.5

    if my_settings.largest:
      my_predict_mask = largest(my_predict_mask)

    my_aff = my_nii.affine
    my_shape = my_nii.shape
    my_data = np.array(my_predict_mask, dtype=np.float32)
    my_data = my_data[0:my_shape[0], 0:my_shape[1], 0:my_shape[2]]
    nib.Nifti1Image(my_data, my_aff).to_filename(output)

    print("finished prediction")

########################################################################################################################
# Utilities                                                                                                            #
########################################################################################################################

class Settings:
    ''' A class defining an object that stores the settings for defining the u-net model'''

    def __init__(self, args=None):
        ''' Create the settings from command line arguments'''

        if args:
            self.epochs = args['epochs']
            self.rate = args['rate']
            self.kernel = args['kernel']
            self.rescale = args['rescale']
            self.batches = args['batches']
            self.channels = args['channels']
            self.raw = args['raw']
            self.largest = args['largest']
            self.labels = args['labels']
            self.augment = args['augment']
        else:
            self.epochs = 40
            self.rate = 0.0001
            self.rescale = 256
            self.kernel = 16
            self.batches = 20 
            self.channels = 1
            self.labels = 1
            self.augment = 0
            self.largest = False
            self.raw = False

def load(model):
    ''' Load a previously trained u-net model'''

    checkpoint = torch.load(model, map_location={'cuda:0':'cpu'})
    my_settings = Settings(checkpoint)
    my_model = UNet2d(my_settings)
    my_model.load_state_dict(checkpoint['state'])
    my_model = nn.Sequential(my_model, nn.Softmax(my_settings.labels))
    return (my_settings, my_model)

def largest(mask):
    ''' extract the largest component '''
    labs, num_lab = snd.label(mask)
    c_size = np.bincount(labs.reshape(-1))
    c_size[0] = 0
    max_ind = c_size.argmax()
    return labs == max_ind

def predict(settings, model, image):

    raw_shape = image.data[0][0].shape
    max_dim = torch.tensor(raw_shape).max()
    factor = float(settings.rescale) / float(max_dim)

    rescaled_image = torch.nn.functional.interpolate(image, scale_factor=factor, \
      mode="trilinear", align_corners = False, recompute_scale_factor=True)
    rescale_shape = rescaled_image.data[0][0].shape
    slice_shape = [settings.channels, settings.rescale, settings.rescale]

    use_gpu = torch.cuda.is_available()

    for my_axis in [0, 1, 2]:

        slice_idx = np.insert(np.delete(np.array([0, 1, 2]), 0), my_axis, 0)
        predict_prob = torch.zeros([rescale_shape[my_axis], settings.rescale, settings.rescale])

        for my_slice in range(rescale_shape[my_axis]):

            image_slice = torch.zeros(slice_shape, dtype=torch.float32)

            for c in range(settings.channels):
                if my_axis == 0:
                    subset = rescaled_image.data[0][c,my_slice,:,:]
                elif my_axis == 1:
                    subset = rescaled_image.data[0][c,:,my_slice,:]
                else:
                    subset = rescaled_image.data[0][c,:,:,my_slice]

                image_slice[c,:subset.shape[0],:subset.shape[1]] = subset

            if use_gpu:
                image_slice = image_slice.cuda()

            predict_slice = model(torch.unsqueeze(Variable(image_slice), 0))
            predict_prob[my_slice,:,:] = predict_slice.data[0][1,:,:]

        if use_gpu:
            predict_prob = predict_prob.cpu()

        predict_prob = predict_prob.permute(slice_idx[0], slice_idx[1], slice_idx[2])
        predict_prob = predict_prob[:rescale_shape[0], :rescale_shape[1], :rescale_shape[2]]
        predict_prob = torch.unsqueeze(predict_prob, 0)
        predict_prob = torch.unsqueeze(predict_prob, 0)
        predict_prob = torch.nn.functional.interpolate(predict_prob, \
          size=raw_shape, mode="trilinear", align_corners=False)
        predict_prob = torch.squeeze(predict_prob)
  
        if my_axis == 0:
            predict_multi = torch.unsqueeze(predict_prob, 3)
        else:
            predict_multi = torch.cat((predict_multi, torch.unsqueeze(predict_prob, 3)), dim=3)

    predict_prob = predict_multi.mean(dim=3)
    predict_prob = predict_prob.numpy()
    
    return predict_prob

def evaluate(model, images, masks, output):
    ''' Evaluate a u-net model with a collection of images and masks '''

    my_settings, my_model = load(model)
    use_gpu = torch.cuda.is_available()
    my_model_on_gpu = next(my_model.parameters()).is_cuda

    if use_gpu:
        if not my_model_on_gpu:
            my_model.cuda()
    else:
        if my_model_on_gpu:
            my_model.cpu()

    dicemap = dict()
    
    dataset = VolumeDataset(my_settings, images, masks)
    loader = DataLoader(dataset=dataset, batch_size=1)
    
    for index, volume in enumerate(loader):
        my_image, my_mask = volume

        my_predict_prob = predict(my_settings, my_model, my_image)
        my_predict_mask = my_predict_prob > 0.5

        if my_settings.largest:
            my_predict_mask = largest(my_predict_mask)

        my_truth_mask = my_mask.data[0].numpy()
        my_overlap = float((my_truth_mask * my_predict_mask).sum())
        my_total = my_truth_mask.sum() + my_predict_mask.sum()
        my_dice = 2.0 * my_overlap / my_total if my_total > 0 else 1

        img_nii = dataset.current_image_nii
        img_path = img_nii.get_filename()
        img_dn, img_file = os.path.split(img_path)
        img_name = os.path.splitext(img_file)[0]
        img_name = os.path.splitext(img_name)[0]

        print("  %s, dice = %g" % (img_name, my_dice))

        if not os.path.exists(output):
            os.mkdir(output)

        img_data = np.array(my_predict_mask, dtype = np.float32)
        img_data = img_data[:img_nii.shape[0],:img_nii.shape[1],:img_nii.shape[2]]
        img = nib.Nifti1Image(img_data, img_nii.affine)
        img.to_filename(os.path.join(output, img_name + ".nii.gz"))

        dicemap[img_name] = my_dice

    return dicemap

########################################################################################################################
# Dataset Handling                                                                                                     #
########################################################################################################################

class VolumeDataset(data.Dataset):
    ''' A torch dataset for individual volumes '''

    def __init__(self, settings, images=None, masks=None):
        super(VolumeDataset, self).__init__()

        self.images = images
        self.masks = masks
        self.raw = settings.raw
        self.channels = settings.channels
        self.cases = sorted(os.listdir(images))

        self.current_image_nii = None
        self.current_mask_nii = None

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        self.current_image_nii = nib.load(os.path.join(self.images, self.cases[index]))
        self.current_mask_nii = nib.load(os.path.join(self.masks, self.cases[index]))

        my_image = np.array(self.current_image_nii.get_data(), dtype=np.float32)
        if len(my_image) == 3:
            my_image = np.expand_dims(my_image, axis=0)
        else:
            my_image = np.transpose(my_image, (3, 0, 1, 2))

        if not self.raw:
            my_image = np.nan_to_num(my_image, nan=0, posinf=0, neginf=0)
            for i in range(self.channels):
                my_image[i] = (my_image[i] - my_image[i].mean()) / (my_image[i].std() + 1e-6)
            my_image[my_image < -10] = 0
            my_image[my_image >  10] = 0

        my_image = torch.from_numpy(my_image)
        my_mask = torch.from_numpy(np.array(self.current_mask_nii.get_data() > 0, dtype=np.int64))

        return (my_image, my_mask)

class SliceDataset(data.Dataset):
    ''' A torch dataset for making image slices from volumetric data'''

    def __init__(self, settings, image, mask):
        super(SliceDataset, self).__init__()

        self.augment = settings.augment
        self.channels = settings.channels
        self.rescale = settings.rescale
        self.raw_shape = image.data[0][0].shape
        self.max_dim = torch.tensor(self.raw_shape).max()
        self.factor = float(settings.rescale) / float(self.max_dim)

        self.image = torch.nn.functional.interpolate(image, scale_factor=self.factor, \
          mode="trilinear", align_corners = False, recompute_scale_factor=True)
        self.rescale_shape = self.image.data[0][0].shape

        self.mask = torch.unsqueeze(mask.float(), 0)
        self.mask = torch.nn.functional.interpolate(self.mask, \
          scale_factor=self.factor, mode="nearest", recompute_scale_factor=True)
        self.mask = torch.squeeze(self.mask.long(), 0)

        self.lenI = self.rescale_shape[0]
        self.lenJ = self.rescale_shape[1]
        self.lenK = self.rescale_shape[2]
        self.lenIJ = self.lenI + self.lenJ

        self.augment_count = self.augment + 1
        self.volume_count = self.image.shape[0]
        self.total_slices = self.lenI + self.lenJ + self.lenK

        self.transform = album.Compose([
            album.ShiftScaleRotate(p=0.75), \
            album.HorizontalFlip(p=0.25),
            album.VerticalFlip(p=0.25)])

        # self.transform = album.Compose([
        #     album.ShiftScaleRotate(p=0.75), \
        #     album.HorizontalFlip(p=0.75),
        #     album.VerticalFlip(p=0.75),
        #     album.GaussNoise(var_limit=0.1, p=0.75),
        #     album.RandomBrightnessContrast(p=0.2)])

        if self.volume_count != 1:
            raise Exception("image volume batch count should be one")

    def __len__(self):
        return self.augment_count * self.total_slices
    
    def __getitem__(self, index):

        slice_index = index % self.total_slices
        augment_index = int((index - slice_index) / self.total_slices)

        image_slice = torch.zeros([self.channels, self.rescale, self.rescale], dtype=torch.float32)
        for c in range(self.channels):
            if slice_index < self.lenI:
                image_raw = self.image.data[0][c,slice_index,:,:]
            elif slice_index < self.lenIJ:
                image_raw = self.image.data[0][c,:,slice_index-self.lenI,:]
            else:
                image_raw = self.image.data[0][c,:,:,slice_index-self.lenIJ]
            image_slice[c,:image_raw.shape[0],:image_raw.shape[1]] = image_raw

        mask_slice = torch.zeros([1, self.rescale, self.rescale], dtype=torch.long)
        if slice_index < self.lenI:
            mask_raw = self.mask.data[0][slice_index,:,:]
        elif slice_index < self.lenIJ:
            mask_raw = self.mask.data[0][:,slice_index-self.lenI,:]
        else:
            mask_raw = self.mask.data[0][:,:,slice_index-self.lenIJ]
        mask_slice[0,:image_raw.shape[0],:image_raw.shape[1]] = mask_raw

        # when augment_index is zero, we pass through the raw data
        if augment_index > 0:
            transformed = self.transform(image=image_slice.numpy(), mask=mask_slice.numpy())
            image_slice = torch.from_numpy(transformed["image"].copy())
            mask_slice = torch.from_numpy(transformed["mask"].copy()).type(torch.LongTensor)
            # note: this process creates a negative stride in the mask, 
            #       so we use copy() to normalize that to work with torch

        return image_slice, mask_slice

########################################################################################################################
# Network Design                                                                                                       #
########################################################################################################################

class UNet2d(nn.Module):
    ''' The u-net torch model definition '''

    def __init__(self, settings):
        ''' setup the network '''

        super(UNet2d, self).__init__()

        conv_blocker = lambda din, dout: nn.Sequential(
            nn.Conv2d(din, dout, kernel_size=3, stride=1, padding=1, bias=True), \
            nn.BatchNorm2d(dout), \
            nn.LeakyReLU(0.1), \
            nn.Conv2d(dout, dout, kernel_size=3, stride=1, padding=1, bias=True), \
            nn.BatchNorm2d(dout), \
            nn.LeakyReLU(0.1))

        conv_upper = lambda din, dout: nn.Sequential(
            nn.ConvTranspose2d(din, dout, kernel_size=4, stride=2, padding=1, bias=True), \
            nn.LeakyReLU(0.1))

        k = settings.kernel

        self.maxpool   = nn.MaxPool2d(2)
        self.contract1 = conv_blocker(settings.channels, k)
        self.contract2 = conv_blocker(k * 1, k * 2)
        self.contract3 = conv_blocker(k * 2, k * 4)
        self.contract4 = conv_blocker(k * 4, k * 8)
        self.contract5 = conv_blocker(k * 8, k * 16)
        self.up5to4    = conv_upper(k * 16, k * 8)
        self.up4to3    = conv_upper(k * 8,  k * 4)
        self.up3to2    = conv_upper(k * 4,  k * 2)
        self.up2to1    = conv_upper(k * 2,  k * 1)
        self.expand4   = conv_blocker(k * 16, k * 8)
        self.expand3   = conv_blocker(k * 8,  k * 4)
        self.expand2   = conv_blocker(k * 4,  k * 2)
        self.expand1   = conv_blocker(k * 2,  k * 1)
        self.output    = nn.Conv2d(k, settings.labels + 1, kernel_size=3, stride=1, padding=1)

        def my_init(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)

        self.apply(my_init)

    def forward(self, x):
        ''' compute the forward pass'''

        contract1out = self.contract1(x)
        contract2out = self.contract2(self.maxpool(contract1out))
        contract3out = self.contract3(self.maxpool(contract2out))
        contract4out = self.contract4(self.maxpool(contract3out))
        contract5out = self.contract5(self.maxpool(contract4out))
        expand5out = contract5out
        expand4out = self.expand4(torch.cat((self.up5to4(expand5out), contract4out), 1))
        expand3out = self.expand3(torch.cat((self.up4to3(expand4out), contract3out), 1))
        expand2out = self.expand2(torch.cat((self.up3to2(expand3out), contract2out), 1))
        expand1out = self.expand1(torch.cat((self.up2to1(expand2out), contract1out), 1))
        out = self.output(expand1out)

        return out

################################################################################
# End
################################################################################
