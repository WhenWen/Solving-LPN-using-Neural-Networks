import torch
import random
from learner import Learner
import tqdm
import argparse
def write_list_to_file(filename, bool_list, type):
    # uint64_t
    with open(filename, type) as f:
        idx = 0
        for number in range((len(bool_list) + 63)// 64):
            cnt = 0
            remaining = min(64, len(bool_list) - number * 64)
            for _ in range(remaining):
                cnt = (cnt << 1) +  bool_list[idx]
                idx += 1
            cnt = cnt << (64 - remaining)
            f.write(str(cnt))
            f.write('\n')








if __name__ == '__main__':
    # Parsing Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--network_path", default = "", type = str, help = "Path to the trained network")
    parser.add_argument("--secret_path", default = "", type = str, help = "Path to the secret")
    parser.add_argument("--pool_data_size", default = 131072, type = int, help = "Size of Gaussian Sample Pool")
    parser.add_argument("--test_data_size", default = 100000, type = int, help = "Size of Testing Number")
    args = parser.parse_args()    
    learner = torch.load(args.network_path)
    secret = torch.load(args.secret_path) # used in gaussian to check whether the final secret is correct
    n = learner.args.data.d
    m =  args.pool_data_size
    m_test = args.test_data_size
    balance = 0 # Counting
    write_list_to_file('secret', secret, 'w')


    with torch.no_grad():
        for query_idx in tqdm.trange(m + m_test):
            query = [random.randint(0,1) for _ in range(n)]
            predict = float(learner(torch.tensor(query[:]).float().cuda())) > 0.5
            query.append(random.randint(0,1))
            predict = (predict+query[-1])%2
            balance += (predict == 1)
            if(query_idx < m):
                query.append(predict)
                name = 'queries'
            else:
                query.append(predict)
                name = 'test_queries'
            if(query_idx == 0 or query_idx == m):
                write_list_to_file(name, query, 'w')
            else:
                write_list_to_file(name, query, 'a')
        print(balance / (m + m_test) )
    