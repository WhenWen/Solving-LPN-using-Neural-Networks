from random import sample
import torch
import torch.nn as nn
import torch.optim as opt
from models import *
from stop_criterion import *
from utils import sample_batch,_log_norm
import sys
import time
import wandb



class Learner(nn.Module):
    def __init__(self, args):
        super(Learner, self).__init__()
        self.args = args
        self.get_model()
        self.get_loss()
        self.get_optimizer()
        self.get_scheduler()
        self.get_stop_criterion()

    def get_model(self):
        args = self.args
        model_type_dict = {"mlp":mlp, "sparse_mlp":sparse_mlp, "cnn":cnn}
        self.network = model_type_dict[args.model.model_type](args)
    
    def get_loss(self):
        args = self.args
        loss_type_dict = {'mse':nn.MSELoss, 'logistic':torch.nn.functional.binary_cross_entropy, 'mae':torch.nn.L1Loss()}
        self.loss = loss_type_dict[args.train.loss]

    
    def get_optimizer(self):
        args = self.args
        lr = float(args.train.optimizer.lr)
        weight_decay = float(args.train.optimizer.weight_decay)
        optimizer_type_dict = {"adam": opt.Adam, "sgd": opt.SGD}
        self.optimizer = optimizer_type_dict[args.train.optimizer.name](self.network.parameters(), lr=lr, weight_decay=weight_decay)
    
    def get_scheduler(self):
        args = self.args
        scheduler_type_dict = {"none": None, 
                               "cosine": torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, args.stop_criterion.step), 
                               "step": torch.optim.lr_scheduler.StepLR(self.optimizer, args.stop_criterion.step//3)}
        self.scheduler = scheduler_type_dict[args.train.scheduler.name]
            
    def get_stop_criterion(self):
        args = self.args
        stop_type_dict = {"step": stop_by_step(args.stop_criterion.step),
                          "time": stop_by_time(args.stop_criterion.time),
                          "acc": stop_by_acc(args.stop_criterion.acc),}
        self.stop_criterion = stop_type_dict[args.stop_criterion.type]

    def forward(self, input):
        return self.network(input)

    def train(self, data, test_data = None):
        # rand is for main_compare.py, test_data is for main_crack.py
        train_loss = []
        eval_loss = []
        train_accuracy = []
        eval_accuracy = []
        best_accuracy = 0
        begin = time.time()
        iter = 0
        while(not self.stop_criterion(duration = time.time() - begin, accuracy = best_accuracy, steps = iter)):
            iter += 1
            self.network.train()
            batch = sample_batch(self.args, data)
            pred = self.forward(batch['input'])
            loss = self.loss(pred, batch['output'])
            if(self.args.train.l1.use == True):
                regularization = 0
                for param in self.network.parameters():
                    regularization += torch.sum(torch.abs(param))
                loss = loss +  float(self.args.train.l1.lamb) * regularization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if(self.scheduler):
                self.scheduler.step()
            logs = _log_norm(self.network)
            logs["iter"] = iter
            if iter % self.args.log_interval == 0:
                accuracy = (pred.round() == batch['output']).float().sum() / batch['output'].shape[0]
                logs["train_loss"] = loss.cpu()
                logs["train_acc"] = accuracy.cpu()
                train_loss.append(loss.cpu())
                train_accuracy.append(accuracy.cpu())
                print("Training loss at iteration {} is {}. Accuracy is {}. {} seconds elapsed".format(iter, loss, accuracy, time.time()-begin))
                if (test_data != None):
                    self.network.eval()
                    loss, accuracy = self.evaluate(test_data)
                    logs["test_loss"] = loss.detach()
                    logs["test_acc"] = accuracy.cpu()
                    eval_loss.append(loss.cpu())
                    eval_accuracy.append(accuracy.cpu())
                    best_accuracy = max(accuracy.cpu(), best_accuracy)
                sys.stdout.flush()
            del batch, pred, loss
            if(self.args.train.l1.use == True):
                del regularization
            if(self.args.use_wandb):
                wandb.log(logs)
        
        return torch.FloatTensor(train_loss), torch.FloatTensor(eval_loss), \
               torch.FloatTensor(train_accuracy), torch.FloatTensor(eval_accuracy)
    
    def evaluate(self, data):
        with torch.no_grad():
            pred = self.forward(data['input'])
            loss = self.loss(pred, data['output'])
            pred = pred.round()
            accuracy = (pred == data['output']).float().sum() / data['output'].shape[0]
            print("Evaluation loss is {}. Accuracy is {}".format(loss, accuracy))
            return loss, accuracy
