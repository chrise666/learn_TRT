import os,shutil,re,random
import numpy as np
import torch
import torch.optim as optim

from tqdm import tqdm
from colorama import Fore

from config import Config
from data import *
from model import *
from model.custom_model import *

from copy import deepcopy

# os.environ["http_proxy"] = "http://127.0.0.1:33210"
# os.environ["https_proxy"] = "http://127.0.0.1:33210"

if __name__ == "__main__":
    cfg = Config()
    os.makedirs(cfg.save,exist_ok=True)

    seed=cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    deterministic=False
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['PYTHONHASHSEED'] = str(seed)

    datasets,classes,classes_num = build_split(cfg)

    metrics={"precision":[]}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for f,(train_set,valid_set) in enumerate(datasets):

        # train_set.resample(schedule={0: 65, 1: 65})
        # valid_set.resample(schedule={0:25,1:25})

        train_loader = DataloaderBase(dataset=train_set, opt=cfg)
        valid_loader = DataloaderBase(dataset=valid_set, opt=cfg)
        # test_loader=train_loader

        model = resnet50(num_classes=len(classes)).to(device)

        # transfer
        # model_t=ConvNext(num_class=len(classes)).to(device)
        # model=torch.load("save/hix/best.pt").to(device)
        # model.backbone.head=deepcopy(model_t.backbone.head)

        # criterion = DiverseExpertLoss(classes_num=classes_num,sade=False,use_label_smoothing=True)
        criterion = nn.CrossEntropyLoss()

        lr = cfg.lr
        weight_decay = cfg.weight_decay
        epoch = cfg.epoch
        optimizer = optim.AdamW(params=model.parameters(),lr=lr,weight_decay=weight_decay)
        # optimizer = Lion(model.parameters(), lr=lr, weight_decay=1e-1)
        # optimizer  = torch.optim.SGD(params=model.parameters(),lr=lr, momentum=0.9, weight_decay=weight_decay)

        # optimizer = optim.AdamW(params=[
        #     {'params': model.backbone.parameters()},
        #     {'params': model.stn.parameters(), 'lr': lr*10.0}
        # ], lr=lr, weight_decay=weight_decay)

        # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epoch,div_factor=50)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, 
        #     mode='max', 
        #     factor=0.5, 
        #     patience=3, 
        #     verbose=True
        # )

        best_precision = 0.0
        for e in range(epoch):
            model.train()

            train_loss = 0.0
            bar=tqdm(enumerate(train_loader),total=len(train_loader))
            for i, (x, y, _) in bar:
                optimizer.zero_grad()
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                # scheduler.step()
                train_loss += loss.item()
                bar.set_postfix(fold=f, epoch=e, loss_average=f"{train_loss / (i + 1):.4f}")

            if (e+1)%cfg.valid_interval==0 or e==0 or (e + 1)==epoch:
                model.eval()

                tp = 0
                count = 0
                bar_valid=tqdm(enumerate(valid_loader), total=len(valid_loader), colour="green")
                for i , (x, y, _) in bar_valid:
                    x, y = x.to(device), y.to(device)
                    count += y.size(0)

                    output = model(x)
                    _, pred = torch.max(output.data, dim=1)
                    tp += (pred == y).sum()
                    precision = torch.true_divide(tp, count).item()
                    bar_valid.set_postfix(fold=f,epoch=e,valid_precision=f"{Fore.GREEN}{precision:.3f}{Fore.RESET}")

                if precision > best_precision:
                    best_precision=max(best_precision,precision)
                    model.to("cpu")
                    torch.save(model.state_dict(), f"{cfg.save}/best.pth")
                    model.to(device)

        metrics["precision"].append(best_precision)
        print(f"{Fore.GREEN}{metrics}{Fore.RESET}")

        # 调整学习率
        # scheduler.step(best_precision)

        # del train_loader
        # del valid_loader
        # del model
        # torch.cuda.empty_cache()

    print(f"{Fore.GREEN}{np.array(metrics['precision']).mean()},{np.array(metrics['precision']).std()}{Fore.RESET}")
