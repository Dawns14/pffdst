import torch
import torch.cuda
from torch import nn
from torch.nn import functional as F
import argparse
import gc
import itertools
import numpy as np
import os
import sys
import time
import pickle
from copy import deepcopy
import copy

from tqdm import tqdm
import warnings

from datasets import get_dataset
import models
from models import all_models, needs_mask, initialize_mask, resnet_models
from thop import clever_format, profile

rng = np.random.default_rng()

def device_list(x):
    if x == 'cpu':
        return [x]
    return [int(y) for y in x.split(',')]


parser = argparse.ArgumentParser()
parser.add_argument('--eta', type=float, help='learning rate', default=0.01)
parser.add_argument('--clients', type=int, help='number of clients per round', default=20)
parser.add_argument('--rounds', type=int, help='number of global rounds', default=400)
parser.add_argument('--epochs', type=int, help='number of local epochs', default=10)
parser.add_argument('--dataset', type=str, choices=('mnist', 'emnist', 'cifar10', 'cifar100','fashion-mnist'),
                    default='mnist', help='Dataset to use')
parser.add_argument('--distribution', type=str, choices=('dirichlet', 'lotteryfl', 'iid'), default='dirichlet',
                    help='how should the dataset be distributed?')
parser.add_argument('--beta', type=float, default=0.1, help='Beta parameter (unbalance rate) for Dirichlet distribution')
parser.add_argument('--total-clients', type=int, help='split the dataset between this many clients. Ignored for EMNIST.', default=200)
parser.add_argument('--min-samples', type=int, default=0, help='minimum number of samples required to allow a client to participate')
parser.add_argument('--samples-per-client', type=int, default=20, help='samples to allocate to each client (per class, for lotteryfl, or per client, for iid)')
parser.add_argument('--prox', type=float, default=0, help='coefficient to proximal term (i.e. in FedProx)')

# Pruning and regrowth options
parser.add_argument('--sparsity', type=float, default=0.1, help='sparsity from 0 to 1')
parser.add_argument('--rate-decay-method', default='cosine', choices=('constant', 'cosine'), help='annealing for readjustment ratio')
parser.add_argument('--rate-decay-end', default=None, type=int, help='round to end annealing')
parser.add_argument('--readjustment-ratio', type=float, default=0.01, help='readjust this many of the weights each time')
parser.add_argument('--pruning-begin', type=int, default=9, help='first epoch number when we should readjust')
parser.add_argument('--pruning-interval', type=int, default=10, help='epochs between readjustments')
parser.add_argument('--rounds-between-readjustments', type=int, default=15, help='rounds between readjustments')
parser.add_argument('--remember-old', default=False, action='store_true', help="remember client's old weights when aggregating missing ones")
parser.add_argument('--sparsity-distribution', default='erk', choices=('uniform', 'er', 'erk'))
parser.add_argument('--final-sparsity', type=float, default=None, help='final sparsity to grow to, from 0 to 1. default is the same as --sparsity')

parser.add_argument('--batch-size', type=int, default=32,
                    help='local client batch size')
parser.add_argument('--l2', default=0.001, type=float, help='L2 regularization strength')
parser.add_argument('--momentum', default=0.9, type=float, help='Local client SGD momentum parameter')
parser.add_argument('--cache-test-set', default=False, action='store_true', help='Load test sets into memory')
parser.add_argument('--cache-test-set-gpu', default=False, action='store_true', help='Load test sets into GPU memory')
parser.add_argument('--test-batches', default=0, type=int, help='Number of minibatches to test on, or 0 for all of them')
parser.add_argument('--eval-every', default=10, type=int, help='Evaluate on test set every N rounds')
parser.add_argument('--device', default='0', type=device_list, help='Device to use for compute. Use "cpu" to force CPU. Otherwise, separate with commas to allow multi-GPU.')
parser.add_argument('--min-votes', default=0, type=int, help='Minimum votes required to keep a weight')
parser.add_argument('--no-eval', default=True, action='store_false', dest='eval')
parser.add_argument('--grasp', default=False, action='store_true')
parser.add_argument('--fp16', default=False, action='store_true', help='upload as fp16')
parser.add_argument('-o', '--outfile', default='output.log', type=argparse.FileType('a', encoding='ascii'))
# 
# freeze and server-readjustment
parser.add_argument('--freeze', default=False, action='store_true')
parser.add_argument('--server-readjustment', default=False, action='store_true')
parser.add_argument('--freeze-grow-sparsity', type=float, default=0, help='start sparsity from 0 to 1')
parser.add_argument('--freeze-grow-method', type=str, choices=('random', 'topk', 'topk-history', 'topk-mix', 'none'), default='random')
parser.add_argument('--freeze-grow-clients', type=int, default=None, help='number of clients for freeze-grow-method (topk, topk-history, topk-mix)')
parser.add_argument('--end-method', default='end', choices=('mid', 'end'), help='annealing for end')
parser.add_argument('--freeze-float-prob', default=False, help='Use Prob grown.', action='store_true')
parser.add_argument('--start-prune', type=int, help='Start adjusting rounds on the server.', default=400)
parser.add_argument('--end-prune', type=int, help='End adjusting rounds on the server.', default=400)
parser.add_argument('--lenet', default=False, help='Use lenet models.', action='store_true')
parser.add_argument('--float-rate', default=0, type=float)
parser.add_argument('--float-prob', default=False, action='store_true')
parser.add_argument('--float-end', default=100, type=int)

args = parser.parse_args()
devices = [torch.device(x) for x in args.device]
args.pid = os.getpid()

if args.lenet:
    args.total_clients = 400
    
if args.freeze and args.server_readjustment:
    args.readjustment_ratio = 0.0
    args.rate_decay_method = 'constant'
    args.freeze_grow_sparsity = 1-args.sparsity
    args.start_prune = args.rounds // 2
    args.end_prune = args.start_prune + args.rounds // 8
    args.float_rate = args.freeze_grow_sparsity // 2
    args.float_end = args.rounds // 8

if args.freeze and ~args.server_readjustment:
    # args.readjustment_ratio = 0.01
    args.freeze_grow_sparsity = 1-args.sparsity
    args.start_prune = args.rounds // 2
    args.end_prune = args.start_prune
    args.float_rate = 0.0
    args.float_end = 0
    args.end_method = 'mid'
    
if ~args.freeze and args.server_readjustment:
    # args.readjustment_ratio = 0.01
    args.float_rate = (1-args.sparsity) / 2
    args.float_end = args.rounds // 4
    args.readjustment_ratio = 0.0
    args.rate_decay_method = 'constant'


if args.rate_decay_end is None:
    args.rate_decay_end = args.rounds // 2
if args.final_sparsity is None:
    args.final_sparsity = args.sparsity
if args.freeze and args.freeze_grow_clients is None:
    args.freeze_grow_clients = args.total_clients

def print2(*arg, **kwargs):
    print(*arg, **kwargs, file=args.outfile)
    print(*arg, **kwargs)

def dprint(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr)

def print_csv_line(**kwargs):
    print2(','.join(str(x) for x in kwargs.values()))

def nan_to_num(x, nan=0, posinf=0, neginf=0):
    x = x.clone()
    x[x != x] = nan
    x[x == -float('inf')] = neginf
    x[x == float('inf')] = posinf
    return x.clone()

def normaliztion(tensor):
    return (tensor - tensor.min()) / ((tensor.max() - tensor.min()) + 1e-12)

def evaluate_global(clients, global_model, progress=False, n_batches=0):
    with torch.no_grad():
        accuracies = {}
        sparsities = {}

        if progress:
            enumerator = tqdm(clients.items())
        else:
            enumerator = clients.items()

        for client_id, client in enumerator:
            accuracies[client_id] = client.test(model=global_model).item()
            sparsities[client_id] = client.sparsity()

    return accuracies, sparsities


def evaluate_local(clients, global_model, progress=False, n_batches=0):

    # we need to perform an update to client's weights.
    with torch.no_grad():
        accuracies = {}
        sparsities = {}

        if progress:
            enumerator = tqdm(clients.items())
        else:
            enumerator = clients.items()

        for client_id, client in enumerator:
            client.reset_weights(global_state=global_model.state_dict(), use_global_mask=True)
            accuracies[client_id] = client.test().item()
            sparsities[client_id] = client.sparsity()

    return accuracies, sparsities

def calculate_grad(clients, global_params, progress=False, n_batches=0, grad_method='topk'):
    grad_dict = {}

    if progress:
        enumerator = tqdm(clients.items())
    else:
        enumerator = clients.items()

    for client_id, client in enumerator:
        if grad_method == 'topk-history':
            client_grad = client.get_grad_history()
        elif grad_method == 'topk-mix':
            client_grad1 = client.get_grad_history()
            client_grad2 = client.train(global_params=global_params, initial_global_params=initial_global_params, calculate_grad=True)
            client_grad = {}
            if client_grad1 == {}:
                client_grad = client_grad2
            else:
                for key in client_grad2.keys():
                    client_grad[key] = (client_grad1[key] + client_grad2[key]) / 2
            
        else:
            client_grad = client.train(global_params=global_params, initial_global_params=initial_global_params, calculate_grad=True)
        
        if client_grad == {}:
            continue

        for key in client_grad.keys():
            if key not in grad_dict:
                grad_dict[key] = torch.abs(deepcopy(client_grad[key]))
            else:
                grad_dict[key] += torch.abs(client_grad[key])
    
    for key in grad_dict.keys():
        grad_dict[key] = grad_dict[key] / len(clients)

    return grad_dict


# Fetch and cache the dataset
dprint('Fetching dataset...')
cache_devices = devices
maxacc = 0
maxacc1 = 0
download_cost_total = 0
dl_mask_total = 0
dl_weight_total = 0
dl_bias_total = 0
ul_mask_total = 0
ul_weight_total = 0
ul_bias_total = 0
download_cost_full_total = 0
upload_cost_total = 0
flops_cost_total = 0
'''
if os.path.isfile(args.dataset + '.pickle'):
    with open(args.dataset + '.pickle', 'rb') as f:
        loaders = pickle.load(f)
else:
    loaders = get_dataset(args.dataset, clients=args.total_clients,
                          batch_size=args.batch_size, devices=cache_devices,
                          min_samples=args.min_samples)
    with open(args.dataset + '.pickle', 'wb') as f:
        pickle.dump(loaders, f)
'''

loaders = get_dataset(args.dataset, clients=args.total_clients, mode=args.distribution,
                      beta=args.beta, batch_size=args.batch_size, devices=cache_devices,
                      min_samples=args.min_samples, samples=args.samples_per_client)

class Client:

    def __init__(self, id, device, train_data, test_data, net=models.MNISTNet,
                 local_epochs=10, learning_rate=0.01, target_sparsity=0.1):
        '''Construct a new client.

        Parameters:
        id : object
            a unique identifier for this client. For EMNIST, this should be
            the actual client ID.
        train_data : iterable of tuples of (x, y)
            a DataLoader or other iterable giving us training samples.
        test_data : iterable of tuples of (x, y)
            a DataLoader or other iterable giving us test samples.
            (we will use this as the validation set.)
        local_epochs : int
            the number of local epochs to train for each round

        Returns: a new client.
        '''

        self.id = id

        self.train_data, self.test_data = train_data, test_data

        self.device = device
        self.net = net(device=self.device).to(self.device)
        initialize_mask(self.net)
        self.criterion = nn.CrossEntropyLoss()

        self.learning_rate = learning_rate
        self.reset_optimizer()

        self.local_epochs = local_epochs
        self.curr_epoch = 0

        # save the initial global params given to us by the server
        # for LTH pruning later.
        self.initial_global_params = None

        # save the grad history for find important weight.
        self.grad_history = {}
        self.grad_history_num = 0
        self.save_grad = (args.freeze and (args.freeze_grow_method == 'topk-history' or args.freeze_grow_method == 'topk-mix'))


    def get_grad_history(self):
        grad_res = deepcopy(self.grad_history)
        for key in grad_res.keys():
            grad_res[key] /= self.grad_history_num
        return grad_res

    def reset_optimizer(self):
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.learning_rate, momentum=args.momentum, weight_decay=args.l2)


    def reset_weights(self, *args, **kwargs):
        return self.net.reset_weights(*args, **kwargs)


    def sparsity(self, *args, **kwargs):
        return self.net.sparsity(*args, **kwargs)


    def train_size(self):
        return sum(len(x) for x in self.train_data)


    def train(self, global_params=None, initial_global_params=None,
              readjustment_ratio=0.5, readjust=False, sparsity=args.sparsity, calculate_grad=False, server_round=0):
        '''Train the client network for a single round.'''

        ul_cost = 0
        dl_cost = 0
        

        client_grad = {}

        if global_params:
            # this is a FedAvg-like algorithm, where we need to reset
            # the client's weights every round
            mask_changed = self.reset_weights(global_state=global_params, use_global_mask=True)

            # Try to reset the optimizer state.
            self.reset_optimizer()

            if mask_changed:
                dl_cost += (1-self.net.sparsity()-args.freeze_grow_sparsity) * self.net.mask_size # need to receive mask
    
            if not self.initial_global_params:
                self.initial_global_params = initial_global_params
                # no DL cost here: we assume that these are transmitted as a random seed
            else:
                # otherwise, there is a DL cost: we need to receive all parameters masked '1' and
                # all parameters that don't have a mask (e.g. biases in this case)
                dl_cost += (1-self.net.sparsity()-args.freeze_grow_sparsity) * self.net.mask_size * 32 + (self.net.param_size - self.net.mask_size * 32)
                
        #pre_training_state = {k: v.clone() for k, v in self.net.state_dict().items()}
        for epoch in range(self.local_epochs):

            self.net.train()

            running_loss = 0.
            for inputs, labels in self.train_data:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
               
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                if args.prox > 0:
                    loss += args.prox / 2. * self.net.proximal_loss(global_params)
                loss.backward()
                self.optimizer.step()

                self.reset_weights() # applies the mask

                running_loss += loss.item()
                
                if calculate_grad and epoch == self.local_epochs - 1:
                    for name, layer in self.net.named_children():
                        for pname, param in layer.named_parameters():
                            if f'{name}.{pname}' not in client_grad:
                                client_grad[f'{name}.{pname}'] = normaliztion(torch.abs(deepcopy(param.grad)))
                            else:
                                client_grad[f'{name}.{pname}'] += normaliztion(torch.abs(param.grad))

            if calculate_grad == False and epoch == (self.local_epochs - 1) and readjust and server_round < args.rate_decay_end:
                prune_sparsity = sparsity + (1 - sparsity) * readjustment_ratio
                # recompute gradient if we used FedProx penalty
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                self.criterion(outputs, labels).backward()

                #  for topk-history
                if self.save_grad:
                    for name, layer in self.net.named_children():
                        for pname, param in layer.named_parameters():
                            if self.grad_history_num == 0:
                                self.grad_history[f'{name}.{pname}'] = normaliztion(torch.abs(deepcopy(param.grad)))
                            else:
                                self.grad_history[f'{name}.{pname}'] += normaliztion(torch.abs(param.grad))
                    self.grad_history_num += 1

                self.net.layer_prune(sparsity=prune_sparsity, sparsity_distribution=args.sparsity_distribution)
                self.net.layer_grow(sparsity=sparsity, sparsity_distribution=args.sparsity_distribution)
                if args.readjustment_ratio > 0:
                    ul_cost += self.net.mask_size
            self.curr_epoch += 1

        # we only need to transmit the masked weights and all biases
        if args.fp16:
            ul_cost += (1-self.net.sparsity()-args.freeze_grow_sparsity) * self.net.mask_size * 16 + (self.net.param_size - self.net.mask_size * 16)
        else:
            ul_cost += (1-self.net.sparsity()-args.freeze_grow_sparsity) * self.net.mask_size * 32 + (self.net.param_size - self.net.mask_size * 32)
        ret = dict(state=self.net.state_dict(), dl_cost=dl_cost, ul_cost=ul_cost)
    
        if calculate_grad:
            for name, layer in self.net.named_children():
                for pname, param in layer.named_parameters():
                    client_grad[f'{name}.{pname}'] /= len(self.train_data)
            return client_grad
        return ret

    def test(self, model=None, n_batches=0):
        '''Evaluate the local model on the local test set.

        model - model to evaluate, or this client's model if None
        n_batches - number of minibatches to test on, or 0 for all of them
        '''
        correct = 0.
        total = 0.

        if model is None:
            model = self.net
            _model = self.net
        else:
            _model = model.to(self.device)

        _model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.test_data):
                if i > n_batches and n_batches > 0:
                    break
                if not args.cache_test_set_gpu:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                outputs = _model(inputs)
                outputs = torch.argmax(outputs, dim=-1)
                correct += sum(labels == outputs)
                total += len(labels)

        # remove copies if needed
        if model is not _model:
            del _model

        return correct / total

# initialize clients
dprint('Initializing clients...')
clients = {}
client_ids = []

new_models = resnet_models
if args.lenet:
    new_models = all_models

for i, (client_id, client_loaders) in tqdm(enumerate(loaders.items())):
    cl = Client(client_id, *client_loaders, net=new_models[args.dataset],
                learning_rate=args.eta, local_epochs=args.epochs,
                target_sparsity=args.sparsity)
    clients[client_id] = cl
    client_ids.append(client_id)
    torch.cuda.empty_cache()

# initialize global model
global_model = new_models[args.dataset](device=devices[0])
initialize_mask(global_model)

# execute grasp on one client if needed
if args.grasp:
    client = clients[client_ids[0]]
    from grasp import grasp
    pruned_net = grasp(client, sparsity=args.sparsity, dataset=args.dataset)
    pruned_masks = {}
    pruned_params = {}
    for cname, ch in pruned_net.named_children():
        for bname, buf in ch.named_buffers():
            if bname == 'weight_mask':
                pruned_masks[cname] = buf.to(device=torch.device(devices[0]), dtype=torch.bool)
        for pname, param in ch.named_parameters():
            pruned_params[(cname, pname)] = param.to(device=torch.device(devices[0]))
    for cname, ch in global_model.named_children():
        for bname, buf in ch.named_buffers():
            if bname == 'weight_mask':
                buf.copy_(pruned_masks[cname])
        for pname, param in ch.named_parameters():
            param.data.copy_(pruned_params[(cname, pname)])
            

else:
    global_model.layer_prune(sparsity=args.sparsity-args.float_rate, sparsity_distribution=args.sparsity_distribution)

initial_global_params = deepcopy(global_model.state_dict())

# we need to accumulate compute/DL/UL costs regardless of round number, resetting only
# when we actually report these numbers
compute_times = np.zeros(len(clients)) # time in seconds taken on client-side for round
download_cost = np.zeros(len(clients))
download_cost_full = np.zeros(len(clients))
upload_cost = np.zeros(len(clients))
flops_cost = np.zeros(len(clients))
rate_decay_start = 0

args.final_sparsity = args.final_sparsity - args.float_rate
args.sparsity = args.sparsity - args.float_rate
tt = copy.deepcopy(args.freeze_grow_sparsity)
args.freeze_grow_sparsity = 0
# for each round t = 1, 2, ... do
for server_round in tqdm(range(args.rounds)):

    if args.float_rate > 0 and server_round <= args.float_end:
        if server_round and server_round % 10 == 0:
            print('global_model.sparsity 1', global_model.sparsity())
            # first prune low param weight, then grown random / grad
            global_model.prune_mask(prune_rate= args.float_rate, prob_method=args.float_prob, start=True)
            print('global_model.sparsity 2', global_model.sparsity())
            if server_round != args.float_end:
                global_model.grown_mask(sparsity=args.float_rate, prob_method=args.float_prob, start=True)
        if server_round == args.float_end:
            args.final_sparsity = args.final_sparsity + args.float_rate
            args.sparsity = args.sparsity + args.float_rate


    if server_round == 0 and args.end_method == 'mid':
        args.rate_decay_end = args.rounds * 0.25
        rate_decay_start = 0

    if server_round == args.rounds * 0.5 and args.end_method == 'mid':
        args.rate_decay_end = args.rounds * 0.75
        rate_decay_start = args.rounds * 0.5
    all_client = False
    if args.freeze and server_round == args.start_prune :
        args.freeze_grow_sparsity = tt
        global_model.load_state_dict(best_params)
        #global_model = torch.load('global_model.pt')
        global_model.freeze()
        if args.freeze_grow_method.startswith('topk'):
            client_indices = rng.choice(list(clients.keys()), size=min(args.freeze_grow_clients, len(list(clients.keys()))), replace=False)
            sample_clients = {client_id:clients[client_id] for client_id in client_indices}
            grad_dict = calculate_grad(clients=sample_clients, global_params=global_model.state_dict(), grad_method=args.freeze_grow_method)
        else:
            grad_dict = None

        if args.freeze_grow_method != 'none':
            global_model.grown(sparsity=args.freeze_grow_sparsity + args.float_rate, grad_dict=grad_dict)
        else:
            all_client = True
        args.sparsity = args.sparsity - args.freeze_grow_sparsity - args.float_rate
        if args.final_sparsity is None:
            args.final_sparsity = args.sparsity
        else:
            args.final_sparsity = args.final_sparsity - args.freeze_grow_sparsity -args.float_rate

    if args.freeze and server_round > args.start_prune and args.start_prune != args.end_prune:
        if server_round <= args.end_prune:
            if server_round % 10 == 0:
                # first prune low param weight, then grown random / grad
                global_model.prune_mask(prune_rate= args.float_rate, prob_method=args.freeze_float_prob, start=False)
                if args.freeze_grow_method.startswith('topk'):
                    client_indices = rng.choice(list(clients.keys()), size=min(args.freeze_grow_clients, len(list(clients.keys()))), replace=False)
                    sample_clients = {client_id:clients[client_id] for client_id in client_indices}
                    grad_dict = calculate_grad(clients=sample_clients, global_params=global_model.state_dict(), grad_method=args.freeze_grow_method)
                else:
                    grad_dict = None

                if args.freeze_grow_method != 'none' and server_round != args.end_prune:
                    # global_model.grown(sparsity=args.float_rate, grad_dict=grad_dict)
                    global_model.grown_mask(sparsity=args.float_rate, grad_dict=grad_dict, prob_method=args.freeze_float_prob, start=False)
                else:
                    all_client = True
        if server_round == args.end_prune:
            # global_model.prune_mask(args.float_rate)
            args.final_sparsity = args.final_sparsity + args.float_rate
            args.sparsity = args.sparsity + args.float_rate


    # sample clients
    if all_client:
        client_indices = list(clients.keys())
        all_client = False
    else:
        client_indices = rng.choice(list(clients.keys()), size=args.clients)

    global_params = global_model.state_dict()
    aggregated_params = {}
    aggregated_params_for_mask = {}
    aggregated_masks = {}
    # set server parameters to 0 in preparation for aggregation,
    for name, param in global_params.items():
        if name.endswith('_mask'):
            continue
        aggregated_params[name] = torch.zeros_like(param, dtype=torch.float, device=devices[0])
        aggregated_params_for_mask[name] = torch.zeros_like(param, dtype=torch.float, device=devices[0])
        if needs_mask(name):
            aggregated_masks[name] = torch.zeros_like(param, device=devices[0])

    # for each client k \in S_t in parallel do
    total_sampled = 0
    round_cost = dict(upload_cost=0, download_cost=0, download_cost_full=0 ,flops_cost=0, dl_mask_cost=0,dl_weight_cost=0,dl_bias_cost=0,ul_mask_cost=0,ul_weight_cost=0,ul_bias_cost=0)
    for client_id in client_indices:
        client = clients[client_id]
        i = client_ids.index(client_id)

        # Local client training.
        t0 = time.process_time()
        
        if args.rate_decay_method == 'cosine':
            readjustment_ratio = global_model._decay(server_round - rate_decay_start, alpha=args.readjustment_ratio, t_end=(args.rate_decay_end - rate_decay_start))
        else:
            readjustment_ratio = args.readjustment_ratio

        readjust = (server_round - rate_decay_start - 1) % args.rounds_between_readjustments == 0 and readjustment_ratio > 0.
        if readjust:
            dprint('readjusting', readjustment_ratio)

        # determine sparsity desired at the end of this round
        # ...via linear interpolation
        if server_round <= args.rate_decay_end:
            round_sparsity = args.sparsity * (args.rate_decay_end - server_round) / (args.rate_decay_end - rate_decay_start) + args.final_sparsity * (server_round - rate_decay_start) / (args.rate_decay_end - rate_decay_start)
        else:
            round_sparsity = args.final_sparsity

        if args.freeze and global_model.freeze_dict is not None and client.net.freeze_dict is None:
            # client.net.freeze_dict = deepcopy(global_model.freeze_dict)
            # freeze weight
            client.net.freeze(global_model.freeze_dict)

        # actually perform training
        train_result = client.train(global_params=global_params, initial_global_params=initial_global_params,
                                    readjustment_ratio=readjustment_ratio,
                                    readjust=readjust, sparsity=round_sparsity, server_round=server_round)
        # make sure freeze weight
        if args.freeze and client.net.freeze_dict is not None:
            for name, param in client.net.freeze_dict.items():
                if not name.endswith('_mask'):
                    continue
                weight_name = name[:-5]
                assert (torch.mul(client.net.freeze_dict[weight_name], param) == torch.mul(client.net.state_dict()[weight_name], param)).all()
                
        cl_params = train_result['state']
        download_cost[i] = train_result['dl_cost']
        upload_cost[i] = train_result['ul_cost']
        t1 = time.process_time()
        compute_times[i] = t1 - t0
        client.net.clear_gradients() # to save memory

        # add this client's params to the aggregate

        cl_weight_params = {}
        cl_mask_params = {}

        # first deduce masks for the received weights
        for name, cl_param in cl_params.items():
            if name.endswith('_orig'):
                name = name[:-5]
            elif name.endswith('_mask'):
                name = name[:-5]
                cl_mask_params[name] = cl_param.to(device=devices[0], copy=True)
                continue

            cl_weight_params[name] = cl_param.to(device=devices[0], copy=True)
            if args.fp16:
                cl_weight_params[name] = cl_weight_params[name].to(torch.bfloat16).to(torch.float)

        # at this point, we have weights and masks (possibly all-ones)
        # for this client. we will proceed by applying the mask and adding
        # the masked received weights to the aggregate, and adding the mask
        # to the aggregate as well.
        for name, cl_param in cl_weight_params.items():
            if name in cl_mask_params:
                # things like weights have masks
                cl_mask = cl_mask_params[name]
                sv_mask = global_params[name + '_mask'].to(devices[0], copy=True)

                # calculate Hamming distance of masks for debugging
                if readjust:
                    dprint(f'{client.id} {name} d_h=', torch.sum(cl_mask ^ sv_mask).item())

                aggregated_params[name].add_(client.train_size() * cl_param * cl_mask)
                aggregated_params_for_mask[name].add_(client.train_size() * cl_param * cl_mask)
                aggregated_masks[name].add_(client.train_size() * cl_mask)
                if args.remember_old:
                    sv_mask[cl_mask] = 0
                    sv_param = global_params[name].to(devices[0], copy=True)

                    aggregated_params_for_mask[name].add_(client.train_size() * sv_param * sv_mask)
                    aggregated_masks[name].add_(client.train_size() * sv_mask)
            else:
                # things like biases don't have masks
                aggregated_params[name].add_(client.train_size() * cl_param)

    # at this point, we have the sum of client parameters
    # in aggregated_params, and the sum of masks in aggregated_masks. We
    # can take the average now by simply dividing...
    for name, param in aggregated_params.items():

        # if this parameter has no associated mask, simply take the average.
        if name not in aggregated_masks:
            aggregated_params[name] /= sum(clients[i].train_size() for i in client_indices)
            continue

        # drop parameters with not enough votes
        aggregated_masks[name] = F.threshold_(aggregated_masks[name], args.min_votes, 0)

        # otherwise, we are taking the weighted average w.r.t. the number of 
        # samples present on each of the clients.
        aggregated_params[name] /= aggregated_masks[name]
        aggregated_params_for_mask[name] /= aggregated_masks[name]
        aggregated_masks[name] /= aggregated_masks[name]

        # it's possible that some weights were pruned by all clients. In this
        # case, we will have divided by zero. Those values have already been
        # pruned out, so the values here are only placeholders.
        aggregated_params[name] = torch.nan_to_num(aggregated_params[name],
                                                   nan=0.0, posinf=0.0, neginf=0.0)
        aggregated_params_for_mask[name] = torch.nan_to_num(aggregated_params[name],
                                                   nan=0.0, posinf=0.0, neginf=0.0)
        aggregated_masks[name] = torch.nan_to_num(aggregated_masks[name],
                                                  nan=0.0, posinf=0.0, neginf=0.0)


    # masks are parameters too!
    for name, mask in aggregated_masks.items():
        aggregated_params[name + '_mask'] = mask
        aggregated_params_for_mask[name + '_mask'] = mask

    global_params = deepcopy(global_model.state_dict())
    # reset global params to aggregated values
    global_model.load_state_dict(aggregated_params_for_mask)
    # golbal mask change . ex conv1.weight_mask
    for name, mask in aggregated_masks.items():
        global_mask_changed = not torch.equal(global_params[name+'_mask'], global_model.state_dict()[name+'_mask'])
        if global_mask_changed:
            print(f'global_mask_changed: {name}_mask', torch.sum(global_params[name+'_mask'] != global_model.state_dict()[name+'_mask']).item(),global_model.state_dict()[name+'_mask'].numel())
    

    if global_model.sparsity() < round_sparsity and args.sparsity != 0:
        # we now have denser networks than we started with at the beginning of
        # the round. reprune on the server to get back to the desired sparsity.
        # we use layer-wise magnitude pruning as before.
        global_model.layer_prune(sparsity=round_sparsity, sparsity_distribution=args.sparsity_distribution)

    # discard old weights and apply new mask
    global_params = global_model.state_dict()
    for name, mask in aggregated_masks.items():
        new_mask = global_params[name + '_mask']
        aggregated_params[name + '_mask'] = new_mask
        aggregated_params[name][~new_mask] = 0
        # make sure freeze weight
        if args.freeze and global_model.freeze_dict is not None:
            aggregated_params[name][global_model.freeze_dict[name+'_mask']] = 0
            aggregated_params[name] += torch.mul(global_model.freeze_dict[name+'_mask'], global_model.freeze_dict[name])
                
    global_model.load_state_dict(aggregated_params)
    
    # make sure freeze weight
    if args.freeze and global_model.freeze_dict is not None:
        for name, param in global_model.freeze_dict.items():
            if not name.endswith('_mask'):
                continue
            weight_name = name[:-5]
            assert (torch.mul(global_model.freeze_dict[weight_name], param) == torch.mul(global_model.state_dict()[weight_name], param)).all()
    
    # evaluate performance
    torch.cuda.empty_cache()
    if server_round % args.eval_every == 0 and args.eval:
        accuracies, sparsities = evaluate_global(clients, global_model, progress=True,
                                                 n_batches=args.test_batches)

    for client_id in clients:
        i = client_ids.index(client_id)
        if server_round == 0:
            clients[client_id].initial_global_params = initial_global_params
    
    average_value = sum(accuracies.values()) / len(accuracies)
    print('server_round:',server_round,',average_value:',average_value)

    if args.freeze and average_value > maxacc and server_round <= args.start_prune:
        best_params = deepcopy(global_model.state_dict())
        #torch.save(global_model,'global_model.pt')
    if average_value > maxacc:
        maxacc = average_value
    if server_round % args.eval_every == 0 and args.eval:
        # clear compute, UL, DL costs
        compute_times[:] = 0
        download_cost[:] = 0
        upload_cost[:] = 0
    print('maxacc',maxacc)


