# *_*coding: utf-8 *_*
# author --Lee--

import os
import torch
import torch.nn as nn

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

# 配置GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据
train_loader = train_data_process()
test_loader = test_data_process()

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

    if "fc" in model_name:
        op_model = model.fc
    else:
        op_model = model

    # 损失
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.AdamW(op_model.parameters(),
                                  lr=opt.lr,
                                  weight_decay=opt.weight_decay)
    lr_scheduler = LRAccuracyScheduler(optimizer, mode='max', patience=5, factor=0.1)
    model_path = str(opt.save_model) + model_name

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    trainer = Trainer(model, train_loader, test_loader, criterion, optimizer, lr_scheduler)
    trainer.train(opt.epochs, model_path)

    re_optimizer = torch.optim.AdamW(remodel.parameters(),
                                     lr=opt.lr / 10,
                                     weight_decay=opt.weight_decay)
    re_lr_scheduler = LRAccuracyScheduler(re_optimizer, mode='max', patience=5, factor=0.1)

    remodel_path = str(opt.save_model) + remodel_name
    if not os.path.exists(remodel_path):
        os.mkdir(remodel_path)
    trainer = Trainer(remodel, train_loader, test_loader, criterion, re_optimizer, re_lr_scheduler)
    trainer.train(opt.epochs, model_path)
