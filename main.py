# *_*coding: utf-8 *_*
# author --Lee--

import os
import torch
import torch.nn as nn
from torchvision import models
import warnings

from opt import parse_opt
from data_load import train_data_process, test_data_process

from utils.EarlyStop import EarlyStopping
from utils.LRAcc import LRAccuracyScheduler
from utils.Trainer import Trainer

from models.bilinear_dense201 import BDense201_fc, BDense201_all
from models.bilinear_dense201 import BDense201_fc_Attention,BDense201_all_Attention
from models.bilinear_dense201 import BDense201_fc_CBAM, BDense201_all_CBAM
from models.bilinear_dense121 import BDense121_fc, BDense121_all
from models.bilinear_dense121 import BDense121_fc_Attention, BDense121_all_Attention
from models.bilinear_dense121 import BDense121_fc_CBAM, BDense121_all_CBAM

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 配置GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据
train_loader = train_data_process()
test_loader = test_data_process()


warnings.filterwarnings("ignore")
if __name__ == '__main__':
    opt = parse_opt()

    if opt.net == "bdense121_fc":
        if opt.attention == "CBAM":
            model = BDense121_fc_CBAM(opt.num_classes).to(device)
            model_name = "Densenet121_fc_CBAM/"
            remodel = BDense121_all_CBAM(opt.num_classes).to(device)
            remodel_name = 'Densenet121_all_CBAM/'
        elif opt.attention == "SelfA":
            model = BDense121_fc_Attention(opt.num_classes).to(device)
            model_name = "Densenet121_fc_SelfA/"
            remodel = BDense121_all_Attention(opt.num_classes).to(device)
            remodel_name = 'Densenet121_all_SelfA/'
        else:
            model = BDense121_fc(opt.num_classes).to(device)
            model_name = "Densenet121_fc/"
            remodel = BDense121_all(opt.num_classes).to(device)
            remodel_name = 'Densenet201_all/'

    elif opt.net == "bdense201_fc":
        if opt.attention == "CBAM":
            model = BDense201_fc_CBAM(opt.num_classes).to(device)
            model_name = "Densenet201_fc_CBAM/"
            remodel = BDense201_all_CBAM(opt.num_classes).to(device)
            remodel_name = 'Densenet201_all_CBAM/'
        elif opt.attention == "SelfA":
            model = BDense201_fc_Attention(opt.num_classes).to(device)
            model_name = "Densenet201_fc_SelfA/"
            remodel = BDense201_all_Attention(opt.num_classes).to(device)
            remodel_name = 'Densenet201_all_SelfA/'
        else:
            model = BDense201_fc(opt.num_classes).to(device)
            model_name = "Densenet201_fc/"
            remodel = BDense201_all(opt.num_classes).to(device)
            remodel_name = 'Densenet201_all/'

    else:
        model = None
        model_name = None
        remodel = None
        remodel_name = None
        print("print error ,please choose the correct net again")
        exit()

    # 损失
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.AdamW(model.fc.parameters(),
                                  lr=opt.lr,
                                  weight_decay=opt.weight_decay)
    lr_scheduler = LRAccuracyScheduler(optimizer, mode='max', patience=3, factor=0.1)
    early_stopping = EarlyStopping(patience=opt.patience/2, delta=0, monitor=opt.monitor)
    model_path = str(opt.save_model) + model_name

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if len(os.listdir(model_path)) != 0:
        initial_checkpoint = model_path+os.listdir(model_path)[0]
        f = torch.load(initial_checkpoint)
        model.load_state_dict(f)
    trainer = Trainer(model, train_loader, test_loader, criterion, optimizer, lr_scheduler, early_stopping)
    trainer.train(int(opt.epochs/10), model_path)     

    # 再次训练
    re_optimizer = torch.optim.AdamW(remodel.parameters(),
                                    lr=opt.lr / 10,
                                    weight_decay=opt.weight_decay)
    re_lr_scheduler = LRAccuracyScheduler(re_optimizer, mode='max', patience=3, factor=0.1)
    re_early_stopping = EarlyStopping(patience=opt.patience, delta=0, monitor=opt.monitor)
    remodel_path = str(opt.save_model) + remodel_name
    if not os.path.exists(remodel_path):
        os.makedirs(remodel_path)

    if len(os.listdir(model_path)) != 0:
        initial_checkpoint = model_path+os.listdir(model_path)[0]
        f = torch.load(initial_checkpoint)
        remodel.load_state_dict(f,strict=False)
    trainer = Trainer(remodel, train_loader, test_loader, criterion, re_optimizer, re_lr_scheduler,re_early_stopping)
    trainer.train(opt.epochs, remodel_path)

    initial_checkpoint = remodel_path+os.listdir(remodel_path)[0]
    test_loss, test_acc = trainer.test_confusion(opt.dataset+'/train', initial_checkpoint)
    print('test_acc = ', test_acc)
    print('test_loss = ', test_loss)
