# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn


from .GRU import BIGRU
from .AGCN import Model as AGCN
from .HCN import HCN

# initilize weight
def weights_init_gru(model):
    with torch.no_grad():
        for child in list(model.children()):
            print(child)
            for param in list(child.parameters()):
                  if param.dim() == 2:
                        nn.init.xavier_uniform_(param)
    print('GRU weights initialization finished!')


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, skeleton_representation, args_bi_gru, args_agcn, args_hcn, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        skeleton_representations: pair of input skeleton representations for training (seq-based_and_graph-based or graph-based_and_image-based or seq-based_and_image-based )
        args_bi_gru: model parameters BIGRU
        args_agcn: model parameters AGCN
        args_hcn: model parameters of HCN
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 16384)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()


        self.K = K
        self.m = m
        self.T = T
        mlp=mlp
        print(" moco parameters",K,m,T,mlp)
        print(skeleton_representation)


        if skeleton_representation=='seq-based_and_graph-based':
               # gru
               self.encoder_q = BIGRU(**args_bi_gru)
               self.encoder_k = BIGRU(**args_bi_gru)
               weights_init_gru(self.encoder_q)
               weights_init_gru(self.encoder_k)

               # AGCN
               self.encoder_r = AGCN(**args_agcn)
               self.encoder_l = AGCN(**args_agcn)

        elif skeleton_representation=='seq-based_and_image-based':
             # gru
             self.encoder_q = GRU_model(**args_bi_gru)
             self.encoder_k = GRU_model(**args_bi_gru)
             weights_init_gru(self.encoder_q)
             weights_init_gru(self.encoder_k)

             # HCN
             self.encoder_r =HCN(**args_hcn)
             self.encoder_l =HCN(**args_hcn)

        elif skeleton_representation=='graph-based_and_image-based':
               # AGCN
               self.encoder_q = AGCN(**args_agcn)
               self.encoder_k = AGCN(**args_agcn)

               # HCN
               self.encoder_r =HCN(**args_hcn)
               self.encoder_l =HCN(**args_hcn)

        #projection heads
        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

            dim_mlp_2 = self.encoder_r.fc.weight.shape[1]
            self.encoder_r.fc = nn.Sequential(nn.Linear(dim_mlp_2, dim_mlp_2), nn.ReLU(), self.encoder_r.fc)
            self.encoder_l.fc = nn.Sequential(nn.Linear(dim_mlp_2, dim_mlp_2), nn.ReLU(), self.encoder_l.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_r, param_l in zip(self.encoder_r.parameters(), self.encoder_l.parameters()):
            param_l.data.copy_(param_r.data)  # initialize
            param_l.requires_grad = False  # not update by gradient

        # create the queue S1 for the one skeleteon-represenation
        self.register_buffer("Queue_S1", torch.randn(dim, K))
        self.Queue_S1= nn.functional.normalize(self.Queue_S1, dim=0)

        self.register_buffer("queue_ptr_S1", torch.zeros(1, dtype=torch.long))

        # create the queue S2 for the other skeleteon-represenation
        self.register_buffer("Queue_S2", torch.randn(dim, K))
        self.Queue_S2= nn.functional.normalize(self.Queue_S2, dim=0)

        self.register_buffer("queue_ptr_S2", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoders
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        for param_r, param_l in zip(self.encoder_r.parameters(), self.encoder_l.parameters()):
            param_l.data = param_l.data * self.m + param_r.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue_Queue_S1(self, keys):

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr_S1)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.Queue_S1[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr_S1[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_Queue_S2(self, keys):

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr_S2)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.Queue_S2[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr_S2[0] = ptr


    def forward(self, input_s1_v1, input_s2_v1, input_s1_v2, input_s2_v2):
        """
        Input:
            input_s1_v1:   s1-based query
            input_s2_v1:   s2-based query
            input_s1_v2:   s1-based key
            input_s2_v2:   s2-based key
        Output:
            logits, targets
        """

        # compute query features for  s1 and  s2 skeleton representations 
        q = self.encoder_q(input_s1_v1)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        r = self.encoder_r(input_s2_v1)  # queries: NxC
        r = nn.functional.normalize(r, dim=1)

        # compute key features for  s1 and  s2  skeleton representations 
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(input_s1_v2)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            l = self.encoder_l(input_s2_v2)  # keys: NxC
            l = nn.functional.normalize(l, dim=1)


        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1, 
        # use  intra-skeleton contrast
        l_pos_seq = torch.einsum('nc,nc->n', [q, l]).unsqueeze(-1) # query-s1 key-s2
        l_pos_graph = torch.einsum('nc,nc->n', [r, k]).unsqueeze(-1) # query-s2 key-s1

        # negative logits: NxK
        # use  intra-skeleton contrast
        l_neg_seq = torch.einsum('nc,ck->nk', [q, self.Queue_S2.clone().detach()]) # query-s1  negative-keys-s2

        l_neg_graph = torch.einsum('nc,ck->nk', [r, self.Queue_S1.clone().detach()]) # query-s2  negative-keys-s1


        # logits: Nx(1+K)
        logits_seq = torch.cat([l_pos_seq, l_neg_seq], dim=1)
        logits_graph = torch.cat([l_pos_graph, l_neg_graph], dim=1)

        # apply temperature
        logits_seq /= self.T
        logits_graph /= self.T

        # labels: positive key indicators
        labels_seq = torch.zeros(logits_seq.shape[0], dtype=torch.long).cuda()
        labels_graph = torch.zeros(logits_graph.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue_Queue_S1(k)
        self._dequeue_and_enqueue_Queue_S2(l)

        return (logits_seq,logits_graph), (labels_seq,labels_graph)
