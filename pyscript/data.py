import os
import tqdm
import torch
import random
import aes_drbg
import argparse
import numpy as np
import yaml
from utils import set_seed, Namespace, flatten


def gen_query(requirelen):
    roundlen = (requirelen//8 + 1)*8
    randomstring = [0]*(roundlen)
    rout = prg.generate(roundlen)
    for _ in range(roundlen//8):
        riter = rout[_]
        for i in range(8):
            randomstring[8*_ + i] = riter%2 
            riter = riter // 2
    randomstring = randomstring[:requirelen]
    randomstring = np.array(randomstring)
    return randomstring

def gen_secret(d, weight):
    cnt = 0
    secret = np.zeros(d)
    while(cnt < weight):
        pos = prg.generate(10)
        pos = int.from_bytes(pos,"little") / (1 << 80)
        pos = int(d * pos)
        cnt += (1 - secret[pos])
        secret[pos] = 1
    return secret


def gen_noise(p):
    noise = prg.generate(10)
    noise = int.from_bytes(noise,"little") / (1 << 80)
    if(noise < p):
        return 1
    else:
        return 0

def generate_dataset(d, weight, noise_rate, num):
    #generate secret
    secret = gen_secret(d, int(d*weight))
    queries = []
    gt = []
    result = []
    for sample in tqdm.trange(num):
        queries.append(gen_query(d))
        gt.append(np.dot(queries[-1],secret)%2)
        noise = gen_noise(noise_rate)
        if(noise):
            result.append(1 - gt[-1])
        else:
            result.append(gt[-1])
    queries = np.stack(queries)
    gt = np.array(gt)
    result = np.array(result)
    challenges = (queries,result)
    answers = (secret,gt)
    return (challenges,answers)

def save_dataset(seed, d, weight, noise_rate, num, repeat):
    for r in range(repeat):
        newdataset = generate_dataset(d, weight, noise_rate, num)
        dirname = "data/{}/{}/{}/{}/{}/{}".format(seed, d, weight, noise_rate, num, r)
        if(not os.path.exists(dirname)):
            os.makedirs(dirname)
        torch.save(newdataset[0],dirname+"/challenge",_use_new_zipfile_serialization = False)
        torch.save(newdataset[1],dirname+"/answer",_use_new_zipfile_serialization = False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", "-c", default = "hyper_config/baseline.yaml", type = str)
    args = parser.parse_args()
    with open(args.config_file, 'r') as f:
        namespace = Namespace(yaml.load(f, Loader=yaml.FullLoader))
        for key, value in namespace.__dict__.items():
            vars(args)[key] = value
        args.config = flatten(namespace.dict)
    seed = args.data.seed
    d = args.data.d
    weight = args.data.s_secret
    repeat = args.data.repeat
    noise_rate = args.data.s_noise
    prg = aes_drbg.AES_DRBG(256)
    prg.instantiate(bytearray([seed]*6))
    set_seed(seed)

    if(args.search):
        for mid in range(args.lower_log, args.upper_log):
            num = int(pow(2,mid/2))*2
            save_dataset(seed, d, weight, noise_rate, num, repeat)
    elif(args.train.num < 0):
        print("Infinite Setting")
    else:
        num = args.train.num + args.test.num
        save_dataset(seed, d, weight, noise_rate, num, repeat)

