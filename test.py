import torch
import torch.nn as nn

from opt import parse_opt
from data_load import train_data_process, test_data_process

from models.bilinear_dense201 import BDense201_all
from models.bilinear_dense201 import BDense201_all_Attention
from models.bilinear_dense201 import BDense201_all_CBAM
from models.bilinear_dense121 import BDense121_all
from models.bilinear_dense121 import BDense121_all_Attention
from models.bilinear_dense121 import BDense121_all_CBAM

from utils.Trainer import Trainer
from utils.EarlyStop import EarlyStopping
from utils.LRAcc import LRAccuracyScheduler


import warnings

# 配置GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载数据
train_loader = train_data_process()
test_loader = test_data_process()

warnings.filterwarnings("ignore")
if __name__ == "__main__":
    opt = parse_opt()
    if opt.net == "bdense121_fc":
        if opt.attention == "CBAM":
            model = BDense121_all_CBAM(opt.num_classes).to(device)
        elif opt.attention == "SelfA":
            model = BDense121_all_Attention(opt.num_classes).to(device)
        else:
            model = BDense121_all(opt.num_classes).to(device)

    elif opt.net == "bdense201_fc":
        if opt.attention == "CBAM":
            model = BDense201_all_CBAM(opt.num_classes).to(device)
        elif opt.attention == "SelfA":
            model = BDense201_all_Attention(opt.num_classes).to(device)
        else:
            model = BDense201_all(opt.num_classes).to(device)
    
    # 损失
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=opt.lr,
                                  weight_decay=opt.weight_decay)
    lr_scheduler = LRAccuracyScheduler(optimizer, mode='max', patience=5, factor=0.1)
    early_stopping = EarlyStopping(patience=10, delta=0, monitor=opt.monitor)

    initial_checkpoint = \
        '/home/hipeson/Bilinear_Densenet/runs/save_model/Densenet201_all/best.pt'

    trainer = Trainer(model, train_loader, test_loader, criterion, optimizer, lr_scheduler, early_stopping)
    test_loss, test_acc = trainer.test_confusion(opt.dataset, initial_checkpoint)
    print('test_acc = ', test_acc)
    print('test_loss = ', test_loss)
