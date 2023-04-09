# This code provides a general framework for learning LPN problem with neural networks
import os
import wandb
import yaml
import torch
import argparse
import sys
import collections
from data import gen_secret
from learner import Learner
from os.path import exists, join
from scipy.stats import ttest_ind
from utils import set_seed, Namespace, flatten, sample_batch



def learn(args, accuracy_list = None):
    args.data.num = args.train.num + args.test.num
    args.data_name = "data_{}_{}_{}_{}_{}_{}".format(args.data.seed, args.data.d, args.data.s_secret, args.data.s_noise, args.data.num, args.data.id)
    args.save_path = join(
                    "raw_results",
                    args.name,
                    args.data_name,
                    str(args.seed))
    if(args.use_wandb):
        wandb.init(project = args.wandb_project_name, 
                    name = os.path.join(args.save_path),
                    entity = args.wandb_entity_name,
                    reinit = True,
                    save_code = True,
                    config = args.config)
    if args.search:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    # get data
    args.data.num = args.train.num + args.test.num
    args.train_data, args.test_data, args.secret = process_data(args) 
    learner = Learner(args).to(args.device)
    try:
        learner.load_state_dict(torch.load(args.resume, map_location=args.device))
    except:
        pass
    save_path = args.save_path
    if not exists(save_path):
        os.makedirs(save_path)
    torch.save(learner.state_dict(), join(save_path, 'initialization.pt'))
    train_loss, eval_loss, train_accuracy, eval_accuracy = learner.train(args.train_data, test_data=args.test_data)
    torch.save(train_loss, join(save_path, 'train_loss.pt'))
    torch.save(train_accuracy, join(save_path, 'train_accuracy.pt'))
    torch.save(eval_loss, join(save_path, 'eval_loss.pt'))
    torch.save(eval_accuracy, join(save_path, 'eval_accuracy.pt'))
    torch.save(learner.state_dict(), join(save_path, 'learner.pt'))
    if(args.use_wandb):
        wandb.finish()
    print("Finish")
    sys.stdout.flush()
    if(accuracy_list is not None):
        accuracy_list.append(eval_accuracy[-1])


def process_data(args):
    if(type(args.train.batch_size) is not int):
        # meaning batch size is a fraction, in this case it corresponds to a fraction of the training set
        args.train.batch_size = int(args.train.batch_size * args.train.num)
    dirname = "data/{}/{}/{}/{}/{}/{}".format(args.data.seed, args.data.d, args.data.s_secret, args.data.s_noise, args.data.num, args.data.id)
    if(args.train.num > 0):
        # finite data
        dataset = torch.load(dirname + "/challenge")
        train_data = collections.defaultdict(list)
        test_data = collections.defaultdict(list)
        output = torch.from_numpy(dataset[1]).unsqueeze(1).float()
        input = torch.from_numpy(dataset[0]).float()
        train_size = args.train.num
        test_size = args.test.num
        train_data['output'] = output[:train_size, :].to(args.device)
        train_data['input'] = input[:train_size, :].to(args.device)
        test_data['output'] = output[train_size:train_size+test_size, :].to(args.device)
        test_data['input'] = input[train_size:train_size+test_size, :].to(args.device)
        print(train_data['input'].size(dim=1))
        assert train_data['input'].size(dim=1) == args.data.d
        assert train_data['input'].size(dim=0) == train_size
        assert test_data['input'].size(dim=0) == test_size
        secret = torch.load(dirname + "/answer")
    else:
        # infinite data
        train_data = None
        size = args.data.d
        index = torch.randperm(size, device = args.device)[:int(args.data.d*args.data.s_secret)]
        secret = torch.zeros(size, device = args.device)
        secret[index] = 1
        args.secret = secret
        test_data = sample_batch(args, None, test = True)
    return train_data, test_data, secret


def t_test(zero_accs, one_accs):
    t_value, p_value = ttest_ind(zero_accs, one_accs, alternative="greater")
    return t_value, p_value


if __name__ == '__main__':
    # Parsing Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", "-c", default = "config/baseline.yaml", type = str)
    parser.add_argument("--seed", default = 0, type = int)
    args = parser.parse_args()
    with open(args.config_file, 'r') as f:
        namespace = Namespace(yaml.load(f, Loader=yaml.FullLoader))
        for key, value in namespace.__dict__.items():
            vars(args)[key] = value
        args.config = flatten(namespace.dict)
    # train
    learn(args)