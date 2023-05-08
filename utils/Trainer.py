import torch
import time
import os
from tqdm import tqdm

from utils.Confusion import ConfusionMatrix


from opt import parse_opt

opt = parse_opt()


class Trainer:
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, lr_scheduler, early_stopping):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.early_stopping = early_stopping

    def train(self, num_epochs, model_path):
        start = time.time()
        model_path = model_path + '/best.pt'
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            total_step = len(self.train_loader)
            # 创建一个进度条，并设置总共的step数量
            loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            for i, (inputs, labels) in loop:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                running_acc = correct / total

                # 更新训练信息
                loop.set_description(f'Epoch [{epoch+1}/{num_epochs}]')
                loop.set_postfix(loss=loss.item(), acc=running_acc)

            train_loss = running_loss / total_step
            train_acc = correct / total

            test_loss, test_acc = self.test()

            self.lr_scheduler.step(test_acc)
            if opt.monitor == 'acc':
                self.early_stopping(test_acc, self.model, model_path)
            else:
                self.early_stopping(test_loss, self.model, model_path)

            print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}'
                  .format(epoch + 1, num_epochs, train_loss, train_acc, test_loss, test_acc))

            if self.early_stopping.early_stop:
                print("Early Stopping")
                break
        end = time.time()
        print('train time cost: {:.5f}'.format(end-start))

    def test(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.test_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss = running_loss / len(self.test_loader)
        test_acc = correct / total

        return test_loss, test_acc

    def test_confusion(self, path, initial_checkpoint):
        f = torch.load(initial_checkpoint)
        self.model.load_state_dict(f)

        # read class_indict
        labels = os.listdir(path)
        confusion = ConfusionMatrix(num_classes=opt.num_classes, labels=labels)

        start = time.time()
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.test_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                confusion.update(predicted.cpu().numpy(), labels.cpu().numpy())
        end = time.time()
        confusion.plot()
        confusion.summary()
        print("test_confusion time cost: {:.5f} sec".format(end - start))

        test_loss = running_loss / len(self.test_loader)
        test_acc = correct / total

        return test_loss, test_acc
