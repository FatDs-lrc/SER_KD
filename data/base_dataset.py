"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import torch
import random
import numpy as np
import torch.utils.data as data
# from PIL import Image
# import torchvision.transforms as transforms
from abc import ABC, abstractmethod

def move_to_cuda(batch):
    if isinstance(batch, torch.Tensor):
        return batch.cuda(non_blocking=True)
    elif isinstance(batch, list):
        new_batch = [move_to_cuda(t) for t in batch]
    elif isinstance(batch, tuple):
        new_batch = tuple(move_to_cuda(t) for t in batch)
    elif isinstance(batch, dict):
        new_batch = {n: move_to_cuda(t) for n, t in batch.items()}
    else:
        return batch
    return new_batch


def record_cuda_stream(batch):
    if isinstance(batch, torch.Tensor):
        batch.record_stream(torch.cuda.current_stream())
    elif isinstance(batch, list) or isinstance(batch, tuple):
        for t in batch:
            record_cuda_stream(t)
    elif isinstance(batch, dict):
        for t in batch.values():
            record_cuda_stream(t)
    else:
        pass

class PrefetchLoader(object):
    """
    overlap compute and cuda data transfer
    (copied and then modified from nvidia apex)
    """
    def __init__(self, loader):
        self.loader = loader
        self.stream = torch.cuda.Stream()

    def __iter__(self):
        loader_it = iter(self.loader)
        self.preload(loader_it)
        batch = self.next(loader_it)
        while batch is not None:
            yield batch
            batch = self.next(loader_it)

    def __len__(self):
        return len(self.loader)

    def preload(self, it):
        try:
            self.batch = next(it)
        except StopIteration:
            self.batch = None
            return
        # if record_stream() doesn't work, another option is to make sure
        # device inputs are created on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input,
        #                                        device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target,
        #                                         device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use
        # by the main stream at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.batch = move_to_cuda(self.batch)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this
            # side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

    def next(self, it):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is not None:
            record_cuda_stream(batch)
        self.preload(it)
        return batch

    def __getattr__(self, name):
        method = self.loader.__getattribute__(name)
        return method


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.manual_collate_fn = False
        # self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}


# def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
#     transform_list = []
#     if grayscale:
#         transform_list.append(transforms.Grayscale(1))
#     if 'resize' in opt.preprocess:
#         osize = [opt.load_size, opt.load_size]
#         transform_list.append(transforms.Resize(osize, method))
#     elif 'scale_width' in opt.preprocess:
#         transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, method)))

#     if 'crop' in opt.preprocess:
#         if params is None:
#             transform_list.append(transforms.RandomCrop(opt.crop_size))
#         else:
#             transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

#     if opt.preprocess == 'none':
#         transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

#     if not opt.no_flip:
#         if params is None:
#             transform_list.append(transforms.RandomHorizontalFlip())
#         elif params['flip']:
#             transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

#     if convert:
#         transform_list += [transforms.ToTensor()]
#         if grayscale:
#             transform_list += [transforms.Normalize((0.5,), (0.5,))]
#         else:
#             transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
#     return transforms.Compose(transform_list)


# def __make_power_2(img, base, method=Image.BICUBIC):
#     ow, oh = img.size
#     h = int(round(oh / base) * base)
#     w = int(round(ow / base) * base)
#     if (h == oh) and (w == ow):
#         return img

#     __print_size_warning(ow, oh, w, h)
#     return img.resize((w, h), method)


# def __scale_width(img, target_width, method=Image.BICUBIC):
#     ow, oh = img.size
#     if (ow == target_width):
#         return img
#     w = target_width
#     h = int(target_width * oh / ow)
#     return img.resize((w, h), method)


# def __crop(img, pos, size):
#     ow, oh = img.size
#     x1, y1 = pos
#     tw = th = size
#     if (ow > tw or oh > th):
#         return img.crop((x1, y1, x1 + tw, y1 + th))
#     return img


# def __flip(img, flip):
#     if flip:
#         return img.transpose(Image.FLIP_LEFT_RIGHT)
#     return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
