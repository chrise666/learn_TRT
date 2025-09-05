import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import torchvision.transforms as T
import albumentations as A
from classify.model.model import *
from PIL import Image

def feature_geo(img):
    h,w,_=img.shape
    return np.array([h,w])

def feature_glcm(img):
    img = np.mean(img,axis=2).astype(np.int)
    glcm = graycomatrix(img, [2, 8, 16], [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4], 256, symmetric=True, normed=True)
    feature=np.array([])
    for a in {'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'}:
        t = graycoprops(glcm, a)
        feature=np.append(feature,t.reshape(-1),axis=0)

    return feature

def feature_rgb(img):
    feature=np.mean(img,axis=(0,1))
    return feature

def feature_fft(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    try:
        fa = np.abs(fshift) #np.log(np.abs(fshift))
        x=np.array([np.max(fa,axis=(0,1,2)),np.min(fa,axis=(0,1,2)),np.mean(fa,axis=(0,1,2))]).astype(np.float16)
    except:
        return None

    if any(np.isinf(x)):
        return None
    return x

class PadToStride(object):
    def __init__(self,stride=32):
        self.stride=stride
        self.value=(114,114,114)

    def __call__(self, img):
        s=np.array(img.shape[:2])
        s_pad=(np.ceil(s/self.stride)*self.stride-s)
        s_pad=s_pad.astype(int)

        top=s_pad[0]//2
        bottom=s_pad[0]-top
        left=s_pad[1]//2
        right=s_pad[1]-left

        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,value=self.value)
        return img

    def __repr__(self):
        return self.__class__.__name__+'()'

class FeatureDeep:
    def __init__(self):
        self.extractor=ConvNext().to("cuda")

        #for mmpretrain models, need different pre transforms
        self.transform=T.Compose([
            # PadToStride(),
            T.ToTensor(),
            # lambda x: x * 255,
            T.Resize(size=(256, 256)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # self.transform=T.Compose([
        #     T.Resize(size=(224, 224)),
        #     T.ToTensor(),
        #     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])

    def __call__(self, image):
        x = cv2.imread(image)
        # x = Image.open(image)
        x=self.transform(x)
        x=x.unsqueeze(0).to("cuda")
        y=self.extractor(x)
        return y.squeeze(0).cpu().detach().numpy()

class Feature:
    def __init__(self):
        pass

    def __call__(self, image):
        img=cv2.imread(image)
        img=cv2.resize(img,(224,224))

        return feature_glcm(img)