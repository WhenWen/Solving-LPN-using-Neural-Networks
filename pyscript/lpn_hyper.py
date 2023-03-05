import os
import re
import wandb
import yaml
import json
import torch
import time
import random
import argparse
import sys
import collections
import numpy as np
from lpn import learn,process_data
from utils import Namespace, flatten
from multiprocessing import Process, Manager, Lock


def reach_accuracy_threshold(accuracy_list):
    max_acc = max(list(accuracy_list))
    eps = 0.05
    ths = 0.5 + np.sqrt(np.log(1/eps) / args.train.num)
    return (max_acc >= ths)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", "-c", default = "hyper_config/baseline.yaml", type = str)
    args = parser.parse_args()
    with open(args.config_file, 'r') as f:
        namespace = Namespace(yaml.load(f, Loader=yaml.FullLoader))
        for key, value in namespace.__dict__.items():
            vars(args)[key] = value
        args.config = flatten(namespace.dict)
    ## The goal is to return the minimal x, such that with 2^{x/2} training data, one out of eight runs return
    lower_log = args.lower_log
    upper_log = args.upper_log
    while(abs(lower_log - upper_log) > 0):
        filelock = Lock()
        mid = (lower_log + upper_log)//2
        args.data.num = int(pow(2,mid/2))*2
        args.train.num = args.data.num//2
        args.test.num = args.data.num//2
        with Manager() as manager:
            accuracy_list = []
            success_dataset_cnt = 0
            for data_id in range(args.num_dataset):
                args.data.id = data_id
                accuracy_list = manager.list()
                for r in range(args.num_parallel//args.max_parallel):
                    print(r)
                    process_list = []
                    for seed in range(min(args.max_parallel,8)):
                        args.seed = r*args.max_parallel + seed
                        args.gpu_id = seed
                        p = Process(target = learn, args = (args, accuracy_list))
                        p.start()
                        process_list.append(p)
                    for p in process_list:
                        p.join()
                success = reach_accuracy_threshold(accuracy_list).int()
                success_dataset_cnt += success
                state_to_name = {0: "Fail", 1: "Success"}
                with open("results/{}".format(args.config_file.split("/")[-1]),"a") as f:
                    f.write("log sample size {} and data id {}: {}\n".format(mid, args.data.id, state_to_name[int(success)]))
        success = (success_dataset_cnt >= args.required_success_rate * args.num_dataset)
        if(success):
            upper_log = mid
        else:
            lower_log = mid + 1
    with open("results/{}".format(args.config_file.split("/")[-1]),"a") as f:
        f.write("minimal log_train_sample: {}\n".format(lower_log))
