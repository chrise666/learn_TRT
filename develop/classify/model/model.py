from torchvision.models import *
import torch.nn as nn
import torch.nn.functional as F
import torch
import random, math
import timm
import numpy as np
import mmpretrain
import kornia
import kornia.augmentation as K
import normflows as nf

class Arc(nn.Module):
    def __init__(self,
                 feature_dim,
                 class_dim,
                 margin=0.4,  # 0.2
                 scale=30.0,  # 30
                 easy_margin=False):
        super().__init__()
        self.feature_dim = feature_dim
        self.class_dim = class_dim
        self.margin = margin
        self.scale = scale
        self.easy_margin = easy_margin
        self.weight = nn.Parameter(torch.FloatTensor(feature_dim, class_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        input_norm = torch.sqrt(torch.sum(torch.square(input), dim=1, keepdim=True))
        input = torch.divide(input, input_norm)

        weight_norm = torch.sqrt(torch.sum(torch.square(self.weight), dim=0, keepdim=True))
        weight = torch.divide(self.weight, weight_norm)

        cos = torch.matmul(input, weight)
        sin = torch.sqrt(1.0 - torch.square(cos) + 1e-6)
        cos_m = math.cos(self.margin)
        sin_m = math.sin(self.margin)
        phi = cos * cos_m - sin * sin_m

        th = math.cos(self.margin) * (-1)
        mm = math.sin(self.margin) * self.margin
        if self.easy_margin:
            phi = self._paddle_where_more_than(cos, 0, phi, cos)
        else:
            phi = self._paddle_where_more_than(cos, th, phi, cos - mm)
        one_hot = torch.nn.functional.one_hot(label, self.class_dim)
        one_hot = torch.squeeze(one_hot, dim=1)
        output = torch.multiply(one_hot, phi) + torch.multiply((1.0 - one_hot), cos)
        output = output * self.scale
        return output

    def _paddle_where_more_than(self, target, limit, x, y):
        mask = (target > limit).float()
        output = torch.multiply(mask, x) + torch.multiply((1.0 - mask), y)
        return output

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, padding=autopad(3))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        att = torch.cat([avgout, maxout], dim=1)
        att = self.sigmoid(self.conv(att))
        return x * att

class ResNet(nn.Module):
    def __init__(self, num_class):
        super(ResNet, self).__init__()
        resnet = resnet50(weights='DEFAULT')
        self.backbone = resnet
        self.proj = nn.Linear(1000, num_class, bias=True)

    def forward(self, x):
        x = self.backbone(x)
        x = self.proj(x)
        return x

class Swin(nn.Module):
    def __init__(self, num_class):
        super(Swin, self).__init__()
        swin = swin_t(weights='DEFAULT')

        # with open('swin_t.txt', 'w') as f:
        #     print(swin, file=f)
        # exit()

        swin.head = nn.Linear(768, num_class, bias=True)
        self.backbone = swin

        # torch.nn.init.xavier_uniform_(self.backbone.head.weight)
        # torch.nn.init.zeros_(self.backbone.head.bias)

        # self.att=SpatialAttention()

    def forward(self, x):
        # x=self.att(x)
        x = self.backbone(x)
        return x

class VIT(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        vit = maxvit_t(weights="DEFAULT")
        vit.classifier[5] = nn.Linear(512, num_class, bias=False)
        self.backbone = vit

    def forward(self, x):
        x = self.backbone(x)
        return x

class MetricModel(nn.Module):
    def __init__(self, dim, num_class):
        super(MetricModel, self).__init__()
        # swin=swin_t(weights='DEFAULT')
        # swin.head=nn.Linear(768,dim,bias=True)
        vit = maxvit_t(weights="DEFAULT")
        vit.classifier[5] = nn.Linear(512, dim, bias=False)
        self.backbone = vit
        # self.att=SpatialAttention()
        self.metric = Arc(dim, num_class)

    def forward(self, x, y):
        # x = self.att(x)
        x = self.backbone(x)
        if self.training:
            return self.metric(x, y)
        else:
            return x

backbones=["convnextv2_tiny.fcmae_ft_in22k_in1k","caformer_s18.sail_in1k","convnextv2_atto.fcmae_ft_in1k",
           "vit_base_patch16_clip_224.laion2b_ft_in1k"]

class ConvNext(nn.Module):
    def __init__(self, num_class=1000):
        super().__init__()
        self.backbone = timm.create_model(backbones[2], pretrained=True, num_classes=num_class)

    def forward(self, x):
        x = self.backbone(x)
        return x

# https://arxiv.org/abs/2304.03977 TotalCodingRate
class TCR(nn.Module):
    def __init__(self, eps=0.01):
        super(TCR, self).__init__()
        self.eps = eps

    def compute_discrimn_loss(self, W):
        p, m = W.shape  # d, B
        I = torch.eye(p, device=W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.

    def forward(self, X):  # B ,d
        return - self.compute_discrimn_loss(X.T)

# import normflows as nf
class Flow(nn.Module):
    def __init__(self,dim=32):
        super().__init__()
        self.backbone = timm.create_model(backbones[3], pretrained=True, num_classes=1000)
        self.proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1000,dim),
        )

        flow_layer = []
        for i in range(8):
            flow_layer.append(nf.flows.AutoregressiveRationalQuadraticSpline(num_input_channels=dim, num_blocks=2, num_hidden_channels=dim*2))
            flow_layer.append(nf.flows.Permute(dim, mode='swap'))

        # flow_layer = []
        # for i in range(8):
        #     param_map = nf.nets.MLP([dim//2, dim*2, dim*2, dim], init_zeros=True)
        #     flow_layer.append(nf.flows.AffineCouplingBlock(param_map))
        #     flow_layer.append(nf.flows.Permute(dim, mode='swap'))

        # self.q0 = nf.distributions.DiagGaussian(dim, trainable=False)
        self.q0 = nf.distributions.GaussianMixture(n_modes=1, dim=dim, trainable=True)
        self.flow = nf.NormalizingFlow(q0=self.q0, flows=flow_layer)

    def forward(self, x):
        x = self.backbone(x)
        x = self.proj(x)

        if self.training:
            loss = self.flow.forward_kld(x)
            return loss
        else:
            log_prob = self.flow.log_prob(x) #torch.exp(·)
            return log_prob

    # can't BP
    def topk(self,x,k=128):
        x = torch.softmax(x,dim=1)
        topk = torch.topk(-x, k=1000-k)
        x[:,topk.indices] = 0.0
        return x

    def train_mode(self,mode="flow"):
        if mode == "flow":
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False

            self.proj.train()
            for p in self.backbone.parameters():
                p.requires_grad = True

            self.flow.train()
            for p in self.flow.parameters():
                p.requires_grad = True
        elif mode == "backbone":
            self.backbone.train()
            for p in self.backbone.parameters():
                p.requires_grad = True

            self.proj.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False

            self.flow.eval()
            for p in self.flow.parameters():
                p.requires_grad = False

# x -> z
class FlowInverse(nn.Module):
    def __init__(self,dim=32):
        super().__init__()
        self.backbone = timm.create_model(backbones[0], pretrained=True, num_classes=1000)
        self.proj=nn.Linear(1000,dim)

        flow_layer = []
        for i in range(8):
            param_map = nf.nets.MLP([dim//2, dim*2, dim*2, dim], init_zeros=True)
            flow_layer.append(nf.flows.AffineCouplingBlock(param_map))
            flow_layer.append(nf.flows.Permute(dim, mode='swap'))

        self.p = nf.distributions.DiagGaussian(dim, trainable = False)
        self.flow = nf.NormalizingFlow(q0 = None,flows = flow_layer,p = self.p)

    def forward(self, x):
        x = self.backbone(x)
        x = self.proj(x)

        if self.training:
            z, log_det = self.flow.forward_and_log_det(x)
            loss= -torch.mean(self.p.log_prob(z)+log_det)
            return loss
        else:
            z, log_det = self.flow.forward_and_log_det(x)
            log_prob = self.p.log_prob(z) + log_det
            return log_prob

    def train_mode(self,mode="flow"):
        if mode == "flow":
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False

            self.proj.train()
            for p in self.backbone.parameters():
                p.requires_grad = True

            self.flow.train()
            for p in self.flow.parameters():
                p.requires_grad = True
        elif mode == "backbone":
            self.backbone.train()
            for p in self.backbone.parameters():
                p.requires_grad = True

            self.proj.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False

            self.flow.eval()
            for p in self.flow.parameters():
                p.requires_grad = False

class GMM(torch.nn.Module):
    def __init__(self, dim=32, n_modes=3):
        super().__init__()

        self.backbone = timm.create_model(backbones[0], pretrained=True, num_classes=1000)
        self.proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1000,dim),
        )

        init_scale = np.sqrt(100.0 / dim)
        self.blend_weight = torch.nn.Parameter( torch.ones(n_modes) )
        self.mean = torch.nn.Parameter(torch.randn(n_modes, dim) * init_scale)
        self.std = torch.nn.Parameter(torch.rand(n_modes, dim) * init_scale)

        self.train_mode()

    def forward(self, x):
        x = self.backbone(x)
        x = self.proj(x)

        blend_weight = torch.distributions.Categorical(torch.nn.functional.relu(self.blend_weight))
        comp = torch.distributions.Independent(torch.distributions.Normal(self.mean, torch.abs(self.std)), 1)
        gmm = torch.distributions.MixtureSameFamily(blend_weight, comp)

        log_prob=gmm.log_prob(x)

        if self.training:
            return -torch.mean(log_prob)
        else:
            return log_prob

    def train_mode(self,mode="gmm"):
        if mode == "gmm":
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False

class ConvNextFeature(nn.Module):
    def __init__(self,):
        super().__init__()
        self.backbone = timm.create_model(backbones[0], pretrained=True, features_only=True)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x=torch.nn.functional.adaptive_avg_pool2d(x,output_size=(1,1)).squeeze((-2,-1))
        return x

class ConvNextMlp(nn.Module):
    def __init__(self, num_class=1000):
        super().__init__()
        self.backbone = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k', pretrained=True, num_classes=512)
        self.encoder = nn.Sequential(nn.Linear(2, 256),nn.ReLU())
        self.mlp = nn.Linear(768,num_class)

    def forward(self, x, z):
        x = self.backbone(x)
        # x=self.head(x)
        return x

class ConvNextDualHead(nn.Module):
    def __init__(self, num_class=1000,aux_num_class=None):
        super().__init__()
        self.backbone = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k', pretrained=True, features_only=True)
        self.gap=nn.AdaptiveAvgPool2d((1,1))
        self.head = nn.Linear(768, num_class, bias=False)
        self.aux_head = nn.Linear(768, aux_num_class, bias=False)

    def forward(self, x, aux_x):
        x = self.backbone(x)[-1]
        x = self.gap(x)
        x = x.view(x.shape[0], -1)
        x = self.head(x)

        if not aux_x==None:
            aux_x = self.backbone(aux_x)[-1]
            aux_x = self.gap(aux_x)
            aux_x = aux_x.view(aux_x.shape[0], -1)
            aux_x = self.aux_head(aux_x)

        return x, aux_x

# Unet -> Transformer
from mmseg.models.backbones import UNet
from mmseg.models.backbones.unet import InterpConv
class ReCoder(nn.Module):
    def __init__(self,num_class,dim=64):
        super().__init__()

        self.ae = UNet(
            in_channels=3,
            base_channels=64,
            num_stages=5,
            strides=(1, 1, 1, 1, 1),
            enc_num_convs=(2, 2, 2, 2, 2),
            dec_num_convs=(2, 2, 2, 2),
            downsamples=(True, True, True, True),
            enc_dilations=(1, 1, 1, 1, 1),
            dec_dilations=(1, 1, 1, 1),
            with_cp=False,
            conv_cfg=None,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            act_cfg=dict(type='ReLU'),
            upsample_cfg=dict(type=InterpConv),
            norm_eval=False
        )

        self.proj = nn.Conv2d(dim,3,kernel_size=1)
        self.head = timm.create_model(backbones[2], pretrained=True, num_classes=num_class)

    def forward(self,x):
        x = self.ae(x)[-1] # b,64,h,w
        x = self.proj(x)
        x = self.head(x)
        return x

    def set_train(self):
        self.ae.train()
        self.proj.train()

        self.head.eval()
        for p in self.head.parameters():
            p.requires_grad = False

        self.head.head.fc.train()
        self.head.head.fc.requires_grad = True

class VGG(nn.Module):
    def __init__(self, num_class=1000):
        super().__init__()
        self.backbone = timm.create_model('vgg16_bn.tv_in1k', pretrained=True, num_classes=num_class)

    def forward(self, x):
        x = self.backbone(x)
        return x

class CV(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = K.ColorJitter(nn.Parameter(torch.tensor([0.0,1.0],device="cuda"))
                               ,nn.Parameter(torch.tensor([0.0,1.0],device="cuda"))
                               ,nn.Parameter(torch.tensor([0.0,1.0],device="cuda"))
                               ,torch.tensor([0.0,0.0],device="cuda")
                               )

    def forward(self,x):
        return self.m(x)

class Sobel(nn.Module):
    def __init__(self, num_class=None):
        super().__init__()

        if num_class:
            self.backbone = timm.create_model(backbones[0], pretrained=True,num_classes=num_class)
        else:
            self.backbone = timm.create_model(backbones[0], pretrained=True)

    def forward(self, x):

        # _,edge=kornia.filters.canny(x, low_threshold=0.1, high_threshold=0.2, kernel_size=(3, 3), sigma=(1, 1),hysteresis=True,eps=1e-6)
        # x=x+edge.repeat(1,3,1,1)

        edge=kornia.filters.sobel(x)
        x=x+edge

        x = self.backbone(x)
        return x

class _Crop(nn.Module):
    def __init__(self, num_class=None):
        super().__init__()
        self.backbone = timm.create_model(backbones[0], pretrained=True,num_classes=8)

        self.proj=nn.Linear(24,num_class)

    def forward(self, x):
        y1=self.backbone(x)
        y2=self.backbone(F.interpolate(x,size=(x.shape[2]//2,x.shape[3]//2)))
        y3 = self.backbone(F.interpolate(x, size=(x.shape[2] // 4, x.shape[3] // 4)))
        return self.proj(torch.cat((y1,y2,y3),dim=1))

class MMPretrain(nn.Module):
    def __init__(self,):
        super().__init__()
        self.model = mmpretrain.get_model(config=r'D:\yc\mmpretrain\benchmark\byol.py'
                                          ,pretrained='D:/save/hix_byol/epoch_160.pth')

    def forward(self, x):
        x = self.model(x)

        return x

class MMPretrainByol(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
        self.model = mmpretrain.get_model(**kwargs)

    def forward(self, x):
        x = self.model(x)

        # byol
        x=x[0]
        x=torch.nn.functional.adaptive_avg_pool2d(x,output_size=(1,1))
        x=x.squeeze((-2,-1))

        return x

from open_clip import create_model_from_pretrained
from PIL import Image
class HF(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model, self.preprocess = create_model_from_pretrained('hf-hub:apple/DFN5B-CLIP-ViT-H-14-378',
                                                                   pretrained=r"D:\yc\DIVA\weight\DFN-ViT-H-14-378.pth"
                                                                   )
        # self.model, self.preprocess = create_model_from_pretrained('hf-hub:timm/ViT-SO400M-14-SigLIP-384')
        # self.model, self.preprocess = create_model_from_pretrained('hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')
        # self.model, self.preprocess = create_model_from_pretrained('hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K')
        # self.model, self.preprocess = create_model_from_pretrained('hf-hub:laion/CLIP-convnext_base_w-laion2B-s13B-b82K')
        
        
    def forward(self, image):
        x=self.preprocess(Image.open(image)).cuda().unsqueeze(0)
        
        y = self.model.encode_image(x)
        # y = F.normalize(y, dim=-1)
        return y

class STN(nn.Module):
    def __init__(self,c):
        super().__init__()

        self.c2=32
        self.feature = nn.Sequential(
            # nn.Conv2d(c, 8, kernel_size=5),
            # nn.MaxPool2d(2, stride=2),
            # nn.ReLU(True),
            nn.Conv2d(c, self.c2, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.AdaptiveMaxPool2d(1)
        )

        # 3 * 2 affine matrix
        self.theta = nn.Sequential(
            nn.Linear(self.c2, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.theta[2].weight.data.zero_()
        self.theta[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        x_ = self.feature(x)
        x_=x_.squeeze(dim=(-1,-2))
        theta = self.theta(x_)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

#not learnable
# class STN2(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#         # affine matrix
#         self.theta=nn.Parameter(torch.tensor([[1, 0, 0],[ 0, 1, 0]], dtype=torch.float))
#
#     def forward(self, x):
#         grid = F.affine_grid(self.theta.data.repeat(x.shape[0],1,1), x.size())
#         x = F.grid_sample(x, grid)
#
#         return x

class NormLinear(nn.Module):
    def __init__(self, c1, c2):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(c1, c2))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

class SADE(nn.Module):
    def __init__(self, num_class=1000):
        super().__init__()
        self.backbone = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k', pretrained=True, num_classes=64)
        self.expert1 = nn.Sequential(nn.Linear(64, num_class, bias=False))
        self.expert2 = nn.Sequential(nn.Linear(64, num_class, bias=False))
        self.expert3 = nn.Sequential(nn.Linear(64, num_class, bias=False))
        self.decide = nn.Sequential(nn.Linear(3 * num_class, num_class, bias=False))
        # self.decide = nn.Sequential(NormLinear(3*num_class,num_class))

    def forward(self, x):
        x = self.backbone(x)
        y1 = self.expert1(x)
        y2 = self.expert2(x)
        y3 = self.expert3(x)
        y = self.decide(torch.cat((y1, y2, y3), dim=1))
        if self.training:
            return [ y, y1, y2, y3]
        else:
            return y
            # return F.softmax(torch.mean(torch.stack((y1,y2,y3),dim=1),dim=1),dim=1)

class AE(nn.Module):
    def __init__(self, num_class=1000):
        super().__init__()
        # self.backbone=convnext_tiny(weights='DEFAULT')
        self.backbone = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k', pretrained=True, num_classes=num_class)
        self.encoder = nn.Sequential(self.backbone, nn.Linear(num_class, 2))
        self.decoder = nn.Sequential(
            nn.Linear(2, num_class, bias=False)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class BCE(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.num_classes=num_classes

    def forward(self,y,label):
        label=F.one_hot(label,num_classes=self.num_classes).to(torch.float)
        return F.binary_cross_entropy_with_logits(y,label,reduction="mean")

from timm.loss import LabelSmoothingCrossEntropy
class LabelSmoothingCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss=LabelSmoothingCrossEntropy()

    def forward(self,x,y):
        return self.loss(x,y)

# SADE loss
class DiverseExpertLoss(nn.Module):
    def __init__(self, classes_num=None, max_m=0.5, s=30, tau=2,sade=True,use_label_smoothing=False):
        super().__init__()
        if use_label_smoothing:
            self.base_loss = LabelSmoothingCE()
        else:
            self.base_loss = BCE(num_classes=len(classes_num))
            # self.base_loss = F.cross_entropy

        self.sade=sade

        prior = np.array(classes_num) / np.sum(classes_num)
        self.prior = torch.tensor(prior).float().cuda()
        self.c = len(classes_num)
        self.s = s
        self.tau = tau

    def inverse_prior(self, prior):
        value, idx0 = torch.sort(prior)
        _, idx1 = torch.sort(idx0)
        idx2 = prior.shape[0] - 1 - idx1  # reverse the order
        inverse_prior = value.index_select(0, idx2)

        return inverse_prior

    def forward(self, logits, target):
        if not self.sade:
            return self.base_loss(logits,target)

        # logit = logits[0]
        expert1_logit = logits[0]
        expert2_logit = logits[1]
        expert3_logit = logits[2]

        # loss0 = self.base_loss(logit, target)

        # Softmax loss for expert 1
        loss1 = self.base_loss(expert1_logit, target)

        # Balanced Softmax loss for expert 2
        expert2_logit = expert2_logit + torch.log(self.prior + 1e-9)
        loss2 = self.base_loss(expert2_logit, target)

        # Inverse Softmax loss for expert 3
        inverse_prior = self.inverse_prior(self.prior)
        expert3_logit = expert3_logit + torch.log(self.prior + 1e-9) - self.tau * torch.log(inverse_prior + 1e-9)
        loss3 = self.base_loss(expert3_logit, target)

        return loss1 + loss2 + loss3
    
    
class GCNFewshot(nn.Module):
    def __init__(self, num_class=1000):
        super().__init__()
        self.backbone = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k', pretrained=True, features_only=True)

    def forward(self, x):
        return x