from collections import defaultdict
import logging
import time

import torch as th
from torch import nn, optim
# from srs.layers.focalloss import *

import torch.nn.functional as F

def evaluate(model, data_loader, prepare_batch, Ks=[20]):
    model.eval()
    results = defaultdict(float)
    max_K = max(Ks)
    num_samples = 0
    with th.no_grad():
        for batch in data_loader:
            # print(batch)
            inputs, labels = prepare_batch(batch)
            # print(inputs)
            # print(len(inputs))
            logits = model(*inputs)
            # logits = logits +logits1
            # print(logits.size())
            batch_size = logits.size(0)
            num_samples += batch_size
            topk = th.topk(logits, k=max_K, sorted=True)[1]
            labels = labels.unsqueeze(-1)
            for K in Ks:
                hit_ranks = th.where(topk[:, :K] == labels)[1] + 1
                hit_ranks = hit_ranks.float().cpu()
                results[f'HR@{K}'] += hit_ranks.numel()
                results[f'MRR@{K}'] += hit_ranks.reciprocal().sum().item()
                results[f'NDCG@{K}'] += th.log2(1 + hit_ranks).reciprocal().sum().item()

    for metric in results:
        results[metric] /= num_samples
    return results


def fix_weight_decay(model, ignore_list=['bias', 'batch_norm']):
    decay = []
    no_decay = []
    logging.debug('ignore weight decay for ' + ', '.join(ignore_list))
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(map(lambda x: x in name, ignore_list)):
            no_decay.append(param)
        else:
            decay.append(param)
    params = [{'params': decay}, {'params': no_decay, 'weight_decay': 0}]
    return params


def print_results(*results_list):
    metrics = list(results_list[0][1].keys())
    logging.warning('Metric\t' + '\t'.join(metrics))
    for name, results in results_list:
        logging.warning(
            name + '\t' +
            '\t'.join([f'{round(results[metric] * 100, 2):.2f}' for metric in metrics])
        )

def compute_kl_loss(p, q, pad_mask=None):
    
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss
    
def compute_kl_loss1(p, q, pad_mask=None):
    
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax((q + p)/2, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax((q + p)/2, dim=-1), reduction='none')
    
    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss


# def js_div(p_output, q_output, get_softmax=True):
#     """
#     Function that measures JS divergence between target and output logits:
#     """
#     KLDivLoss = nn.KLDivLoss(reduction='batchmean')
#     if get_softmax:
#         p_output = F.softmax(p_output,dim=-1)
#         q_output = F.softmax(q_output,dim=-1)
#     log_mean_output = ((p_output + q_output )/2).log()
#     return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2

class TrainRunner:
    def __init__(
        self,
        train_loader,
        valid_loader,
        test_loader,
        model,
        prepare_batch,
        Ks=[20],
        lr=1e-3,
        weight_decay=0,
        ignore_list=None,
        patience=2,
        OTF=False,
        **kwargs,
    ):
        self.model = model
        if weight_decay > 0:
            if ignore_list is not None:
                params = fix_weight_decay(model, ignore_list)
            else:
                params = model.parameters()
        # self.optimizer = optim.Adam(params, lr=lr)
        self.optimizer = optim.AdamW(params, lr=lr, amsgrad=True)
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.prepare_batch = prepare_batch
        self.Ks = Ks
        self.epoch = 0
        self.batch = 0
        self.patience = patience if patience > 0 else 2
        self.precompute = hasattr(model, 'KGE_layer') and not OTF

    def train(self, epochs, beta, log_interval=100):
        best_results = defaultdict(float)
        report_results = defaultdict(float)
        bad_counter = 0
        t = time.time()
        mean_loss = 0
        flag = True
        # torch.autograd.set_detect_anomaly(True)
        for epoch in range(epochs):
            self.model.train()
            train_ts = time.time()
            flag = not flag
            for batch in self.train_loader:
 
                inputs,labels = self.prepare_batch(batch)

                self.optimizer.zero_grad()

                # keep dropout and forward twice∂
                logits = self.model(*inputs)
                # loss  =self.criterion(logits, labels)
                logits1 = self.model(*inputs)
                # cross entropy loss for classifier
                ce_loss = 0.5 * (self.criterion(logits, labels) + self.criterion(logits1, labels))

                kl_loss = compute_kl_loss1(logits, logits1)

                # carefully choose hyper-parameters
                loss = ce_loss + beta * kl_loss
                th.autograd.set_detect_anomaly(True)
                loss.backward()
                self.optimizer.step()
                mean_loss += loss.item() / log_interval
                if self.batch > 0 and self.batch % log_interval == 0:
                    logging.info(
                        f'Batch {self.batch}: Loss = {mean_loss:.4f}, Elapsed Time = {time.time() - t:.2f}s'
                    )
                    t = time.time()
                    mean_loss = 0
                self.batch += 1
                # if self.batch == 500: break
            eval_ts = time.time()
            logging.debug(
                f'Training time per {log_interval} batches: '
                f'{(eval_ts - train_ts) / len(self.train_loader) * log_interval:.2f}s'
            )
            if self.precompute:
                ts = time.time()
                self.model.precompute_KG_embeddings()
                te = time.time()
                logging.debug(f'Precomuting KG embeddings took {te - ts:.2f}s')

            ts = time.time()
            valid_results = evaluate(
                self.model, self.valid_loader, self.prepare_batch, self.Ks
            )
            test_results = evaluate(
                self.model, self.test_loader, self.prepare_batch, self.Ks
            )
            if self.precompute:
                # release precomputed KG embeddings
                self.model.KG_embeddings = None
                th.cuda.empty_cache()
            te = time.time()
            num_batches = len(self.valid_loader) + len(self.test_loader)
            logging.debug(
                f'Evaluation time per {log_interval} batches: '
                f'{(te - ts) / num_batches * log_interval:.2f}s'
            )

            logging.warning(f'Epoch {self.epoch}:')
            print_results(('Valid', valid_results), ('Test', test_results))

            any_better_result = False
            for metric in valid_results:
                if valid_results[metric] > best_results[metric]:
                    best_results[metric] = valid_results[metric]
                    report_results[metric] = test_results[metric]
                    any_better_result = True

            if any_better_result:
                bad_counter = 0
            else:
                bad_counter += 1
                if bad_counter == self.patience:
                    break
            self.epoch += 1
            eval_te = time.time()
            t += eval_te - eval_ts
        print_results(('Report', report_results))
        return report_results
