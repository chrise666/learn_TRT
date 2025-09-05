import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import albumentations as A
import torchvision.transforms as T
import os, cv2, re
from sklearn.model_selection import train_test_split, KFold
from PIL import Image
import numpy as np

# 数据准备
def build_sample(opt,path=None):
    if path==None:
        data_root=opt.data_root
    else:
        data_root=path

    classes = [t for t in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, t))]
    print(classes)
    samples = []
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if re.findall("\.(png|jpg|bmp)", file):
                c = os.path.basename(root)
                samples.append({"file": os.path.join(root, file), "class": c, "id": classes.index(c)})

    # num of each class, pour SAGE loss
    classes_num = [0] * len(classes)
    for s in samples:
        classes_num[s["id"]] += 1

    return samples, classes, classes_num

def build_split(opt):
    Data = DatasetImage
    samples, classes, classes_num = build_sample(opt)
    seed = opt.split_seed

    datasets = []
    if opt.fold > 1:
        kf = KFold(n_splits=opt.fold, random_state=seed, shuffle=True)
        for (train_index, test_index) in kf.split(samples):
            train_sample = [samples[i] for i in train_index]
            test_sample = [samples[i] for i in test_index]
            train_set = Data(opt=opt, train=True, samples=train_sample)
            test_set = Data(opt=opt, train=False, samples=test_sample)
            datasets.append((train_set, test_set))
    else:
        if not opt.valid_frac == 0.0:
            train_sample, test_sample = train_test_split(samples, random_state=seed, test_size=opt.valid_frac)
        else:
            train_sample = samples
            test_sample = samples
        train_set = Data(opt=opt, train=True, samples=train_sample)
        test_set = Data(opt=opt, train=False, samples=test_sample)
        datasets.append((train_set, test_set))

    return datasets, classes, classes_num

def build_split_ml(opt):
    _samples, classes, classes_num = build_sample(opt)
    seed = opt.split_seed

    samples = []
    if opt.fold > 1:
        kf = KFold(n_splits=opt.fold, random_state=seed, shuffle=True)
        for (train_index, test_index) in kf.split(_samples):
            train_sample = [_samples[i] for i in train_index]
            test_sample = [_samples[i] for i in test_index]
            samples.append((train_sample, test_sample))
    else:
        if not opt.valid_frac == 0.0:
            train_sample, test_sample = train_test_split(_samples, random_state=seed, test_size=opt.valid_frac)
        else:
            train_sample = _samples
            test_sample = []
        samples.append((train_sample, test_sample))

    return samples, classes, classes_num

# 数据增强
class Augment:
    def __init__(self, opt, test_aug):
        self.album = A.Compose(test_aug+[
            A.RandomScale(p=1.0, scale_limit=(-0.3, 0.3)),
            A.PadIfNeeded(min_height=opt.image_size[0], min_width=opt.image_size[1], fill=0),
            A.RandomCrop(height=opt.image_size[0], width=opt.image_size[1]),
        ])

    def __call__(self, **kwargs):
        return self.album(**kwargs)

class AugmentBlur:
    def __init__(self, opt):
        self.album = A.Compose([
            A.Resize(height=opt.image_size[0], width=opt.image_size[1]),
            A.RandomScale(p=0.5, scale_limit=(-0.5, 0.5)),
            A.PadIfNeeded(min_height=opt.image_size[0], min_width=opt.image_size[1]),
            A.RandomCrop(height=opt.image_size[0], width=opt.image_size[1]),
            A.RandomBrightnessContrast(p=1.0, brightness_limit=0.0, contrast_limit=(-0.3, 0.3))
        ])

    def __call__(self, **kwargs):
        return self.album(**kwargs)

class AugmentStrong:
    def __init__(self, opt):
        self.album = A.Compose([
            A.Resize(height=opt.image_size[0], width=opt.image_size[1])
            , A.ColorJitter(p=0.5)
            # , A.Cutout(max_h_size=56, max_w_size=56, p=0.5, fill_value=114)
            , A.RandomScale(p=1.0, scale_limit=(-0.3, 0.3))
            , A.PadIfNeeded(min_height=opt.image_size[0], min_width=opt.image_size[1], value=114)
            , A.RandomCrop(height=opt.image_size[0], width=opt.image_size[1])
        ])

    def __call__(self, **kwargs):
        return self.album(**kwargs)

# 数据集与数据加载器
class DatasetImage(Dataset):
    def __init__(self, opt, train=True, samples=None, classes=None, UDA=False, data_root=None):
        self.data_root = opt.data_root
        self.samples = samples
        self.classes = classes
        self.return_meta = opt.return_meta
        if not self.samples:
            self.samples, self.classes, self.classes_num = build_sample(opt, path=data_root)

        self.train = train
        self.UDA = UDA

        self.test_aug = [
            # A.ToGray(),
            A.Resize(height=opt.image_size[0], width=opt.image_size[1]),
            # A.LongestMaxSize(max_size=opt.image_size[0]),
            # A.PadIfNeeded(min_height=opt.image_size[0], min_width=opt.image_size[1], value=114),
        ]

        if self.train:
            self.album = Augment(opt,test_aug=self.test_aug)
            # self.album2 = AugmentStrong(opt)
        else:
            self.album = A.Compose(self.test_aug)

        self.final_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5,], std=[0.5,])  # 需要验证
        ])

    def __getitem__(self, index):
        data = self.samples[index]
        return data

    def __len__(self):
        return len(self.samples)

    def collate(self, batch):
        '''
        batch: [{"file":image absolute path,"id":class id},{}...]
        '''
        if self.train and self.UDA:
            return self.collate_uda(batch)

        X, Y = [], []
        for b in batch:
            # x = cv2.imread(b["file"])
            x = Image.open(b["file"]).convert("RGB")
            x = np.array(x)
            if self.album:
                x = self.album(image=x)["image"]
            x = self.final_transform(x)
            X.append(x)

            y = b["id"]
            Y.append(y)

        if not self.return_meta:
            return torch.stack(X, dim=0), torch.LongTensor(Y)
        else:
            return torch.stack(X, dim=0), torch.LongTensor(Y), batch

    def collate_uda(self, batch):
        '''
        batch: [{"file":image absolute path,"id":class id},{}...]
        '''
        X1, X2, Y = [], [], []
        for b in batch:
            x = cv2.imread(b["file"])
            x1 = self.album(image=x)["image"]
            x2 = self.album2(image=x)["image"]
            x1 = self.final_transform(x1)
            x2 = self.final_transform(x2)

            X1.append(x1)
            X2.append(x2)

            y = b["id"]
            Y.append(y)

        if not self.return_meta:
            return torch.stack(X1, dim=0), torch.stack(X2, dim=0), torch.LongTensor(Y)
        else:
            return torch.stack(X1, dim=0), torch.stack(X2, dim=0), torch.LongTensor(Y),batch

    def resample(self, schedule):
        '''
        schedule: class_id and resample to how many? say {0:10,1:10...}
        '''
        new_sample = []
        for c in self.classes.values():
            old_samples = [t for t in self.samples if t["id"] == c]
            require = schedule[c]
            rest = require
            while rest > 0:
                k = min(len(old_samples), rest)
                new_sample += random.sample(old_samples, k)
                rest -= k
        self.samples = new_sample

class DataloaderBase(DataLoader):
    def __init__(self, dataset, opt):
        super(DataloaderBase, self).__init__(dataset=dataset
                                             , shuffle=True
                                             , batch_size=opt.batch_size
                                             , num_workers=opt.num_workers
                                             , collate_fn=dataset.collate
                                             , pin_memory=False
                                             , persistent_workers=True
                                             , drop_last=False
                                             )
