import collections
import torch
import random
import numpy
import re
def gen_noise(p,num,device):
    noise = torch.rand([num,1],device=device)
    return (noise < p)

def sample_batch(args, data, test = False):
    batch = collections.defaultdict(list)
    if(data is not None):
        size = data['input'].shape[0]
        index = torch.randint(0, size, (args.train.batch_size,), device = args.device)
        batch['input'] = data['input'][index]
        batch['output'] = data['output'][index]
    else:
        # sample fresh
        batch = collections.defaultdict(list)
        if(test):
            num = args.test.num
        else:
            num = args.train.batch_size
        queries = torch.randint(0,2, [num, args.data.d], device= args.device).float()
        secret = args.secret.float()
        if(not args.test.clean or not test):
            result = (torch.mm(queries,secret.unsqueeze(1))+ gen_noise(args.data.s_noise, num, args.device))%2
        else:
            result = (torch.mm(queries,secret.unsqueeze(1)))%2
        batch['input'] = queries
        batch['output'] = result.to(args.device)
    return batch


def abbrv_module_name(name):
    name = name.replace('module.','')
    name = name.replace('bias','b')
    name = name.replace('weight','w')
    name = name.replace('features','L')        
    split_name = name.split('.')
    name = '.'.join(split_name[:])
    return name

def _log_norm(model):
    logs = {'norm/total':0}
    for n,p in model.named_parameters():
        norm = p.norm().detach()
        logs['norm/total'] += norm
        logs['norm/'+ abbrv_module_name(n)] = norm
    logs['norm/total'] = torch.sqrt(logs['norm/total'])
    return logs

class Namespace(object):
    def __init__(self, somedict):
        self.dict = {}
        for key, value in somedict.items():
            assert isinstance(key, str) and re.match("[A-Za-z_-]", key)
            if isinstance(value, dict):
                self.__dict__[key] = Namespace(value)
                self.dict[key] = self.__dict__[key].dict
            else:
                self.__dict__[key] = value
                self.dict[key] = value
    
    def __getattr__(self, attribute):

        raise AttributeError(f"Can not find {attribute} in namespace. Please write {attribute} in your config file(xxx.yaml)!")


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)


class cos(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return (torch.cos(input)+1)/2

def flatten(somedict):
    newdict = {}
    for key, value in somedict.items():
        if(isinstance(value, dict)):
            subdict = flatten(value)
            for sub_key, sub_value in subdict.items():
                newdict[key+'.'+sub_key] = sub_value
        else:
            newdict[key] = value
    return newdict


def get_subfield(args, name):
    try:
        return args.name
    except:
        args.name = None
        return args.name