import os
import argparse

ROOT = os.getcwd()


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=4, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--num-classes', type=int, default=46)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--net', type=str, default='bdense201_fc', help='choose which net to train')
    parser.add_argument('--attention', type=str, default='CBAM', help='choose which attention to retrain')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight-decay', type=float, default=0.00001)
    parser.add_argument('--dataset', default=ROOT+"/exp_data/")
    parser.add_argument('--save-model', default=ROOT + '/runs/save_model/', help='save to project/name')
    parser.add_argument('--patience',type=int,default=10)
    parser.add_argument('--monitor', type=str, default='acc')

    return parser.parse_known_args()[0] if known else parser.parse_args()
