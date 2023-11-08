import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchlight import import_class
import random
from .skeletonAdaIN import AdaIN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
class CStrCRL(nn.Module):
    def __init__(self, base_encoder=None, pretrain=True, feature_dim=128, queue_size=32768,
                 momentum=0.999, Temperature=0.07, mlp=True, in_channels=3, hidden_channels=64,
                 hidden_dim=256, num_class=60, dropout=0.5,dropout_graph=0.1,add_graph=0.05,
                 graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
                 edge_importance_weighting=True, transformer=False, rep_ratio=0,doublequeue=False,**kwargs):
        """
        K: queue size; number of negative keys (default: 32768)
        m: momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super().__init__()
        base_encoder = import_class(base_encoder)
        self.pretrain = pretrain
        self.q_tem_feature_bank=[]

        if pretrain:
            self.register_parameter('mask_param', nn.Parameter(torch.zeros((1,))) )
        if not self.pretrain:
            if transformer:
                self.encoder_q = base_encoder(num_class= num_class, **kwargs)
            else:
                self.encoder_q = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                          hidden_dim=hidden_dim, num_class=num_class,
                                          dropout=dropout, graph_args=graph_args,
                                          edge_importance_weighting=edge_importance_weighting,
                                          dropout_graph=dropout_graph,add_graph=add_graph,
                                          **kwargs)
        else:
            self.K = queue_size
            self.m = momentum
            self.T = Temperature
            print('dropout_graph=',dropout_graph,'add_graph=',add_graph)
            if transformer:
                self.encoder_q = base_encoder(num_class= feature_dim, **kwargs)
                self.encoder_k = base_encoder(num_class= feature_dim, **kwargs)
            else:
                self.encoder_q = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                            hidden_dim=hidden_dim, num_class=feature_dim,
                                            dropout=dropout, graph_args=graph_args,
                                            edge_importance_weighting=edge_importance_weighting,
                                            dropout_graph=dropout_graph,add_graph=add_graph,
                                            **kwargs)
                self.encoder_k = base_encoder(in_channels=in_channels, hidden_channels=hidden_channels,
                                            hidden_dim=hidden_dim, num_class=feature_dim,
                                            dropout=dropout, graph_args=graph_args,
                                            edge_importance_weighting=edge_importance_weighting,
                                            dropout_graph=dropout_graph,add_graph=add_graph,
                                            **kwargs)

            if mlp:  # hack: brute-force replacement
                dim_mlp = self.encoder_q.fc.weight.shape[1]
                self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                  nn.ReLU(),
                                                  self.encoder_q.fc)
                self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                  nn.ReLU(),
                                                  self.encoder_k.fc)
            
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient

            self.register_buffer("queue", torch.randn(feature_dim, queue_size)) #128 32768
            self.queue = F.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        gpu_index = keys.device.index
        self.queue[:, (ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1))] = keys.T
    
    @torch.no_grad()
    def update_ptr(self, batch_size):
        assert self.K % batch_size == 0  # for simplicity
        self.queue_ptr[0] = (self.queue_ptr[0] + batch_size) % self.K

    def forward(self, im_q_1, im_q, im_q_tem, im_k=None, nnm=False, topk=1,KNN=False,return_feat=False):
        """
        random dropout and add edges for adj graph
        Input:
            im_q: a batch of query sequences
            im_k: a batch of key sequences
            im_q_1: a batch of extremely augmented query sequences
        """
        return self.encoder_q(im_q, KNN=KNN, return_feat=return_feat)
        # return self.encoder_q(im_q_tem, KNN=KNN, return_feat=return_feat)
        # return (self.encoder_q(im_q, KNN=KNN, return_feat=return_feat) + self.encoder_q(im_q_tem, KNN=KNN, return_feat=return_feat))/2
    
    def forward_pretrain_wmask(self, im_q_1, im_q_2, im_q_tem, im_q, im_k=None, im_k_str=None, nnm=False, topk=1, mask=None, tsne_features=False, epoch=0):
        """
        Only mask augmentation is applied in the third branch 
        Input:
            im_q: a batch of query sequences
            im_q_1: a batch of query sequences corresponding to the normal augmented branch
            im_q_2: a batch of query sequences corresponding to the mask augmented branch
            im_k: a batch of key sequences
        """
        if mask != None:
            im_q_2 = im_q_2.permute(0,2,4,3,1)# NTMVC
            im_q_2 = im_q_2*mask
            im_q_2 [mask==0] = self.mask_param # Learnable param
            im_q_2 = im_q_2.permute(0,4,1,3,2)#NCTVM
        if nnm:
            return self.nearest_neighbors_mining_mutalddm_wmask(im_q, im_k, im_q_1, im_q_2, im_q_tem, topk, im_k_str=im_k_str)

        if not self.pretrain:
            return self.encoder_q(im_q)

        # Obtain the basic augmented feature
        q = self.encoder_q(im_q)  # NxC
        q_tem = self.encoder_q(im_q_tem) # test whether is N*C
        #Obtain the features in other branches
        #drop_graph=False -> No drop edges applied
        if mask!=None:
            q_1 = self.encoder_q(im_q_1, drop_graph=False)  # NxC
            q_2 = self.encoder_q(im_q_2, drop_graph=False)  # NxC

        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_tem = F.normalize(q_tem, dim=1)
        q_1 = F.normalize(q_1, dim=1)
        q_2 = F.normalize(q_2, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)        

        # Compute logits of normally augmented query using Einstein sum
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        # apply temperature
        logits /= self.T
        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # temporal q
        l_pos_t = torch.einsum('nc,nc->n',[q_tem, k]).unsqueeze(-1)
        l_neg_t = torch.einsum('nc,ck->nk',[q_tem, self.queue.clone().detach()])
        logits_tem = torch.cat([l_pos_t, l_neg_t], dim=1)
        logits_tem /= self.T
        # logits_tem_ddm = logits_tem.clone()
        # logits_tem = torch.softmax(logits_tem, dim=1)
        # Compute logits_1 corresponding to the normal augmentation
        # positive logits: Nx1
        l_pos_e = torch.einsum('nc,nc->n', [q_1, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_e = torch.einsum('nc,ck->nk', [q_1, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_1 = torch.cat([l_pos_e, l_neg_e], dim=1)
        # apply temperature
        logits_1 /= self.T
        logits_1_ddm = logits_1.clone()
        logits_1 = torch.softmax(logits_1, dim=1)

        # Compute logits_2 corresponding to the mask augmentation
        # positive logits: Nx1
        l_pos_ed = torch.einsum('nc,nc->n', [q_2, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_ed = torch.einsum('nc,ck->nk', [q_2, self.queue.clone().detach()])
        # logits: Nx(1+K)
        logits_2 = torch.cat([l_pos_ed, l_neg_ed], dim=1)
        # apply temperature
        logits_2 /= self.T
        logits_2 = torch.softmax(logits_2, dim=1)

        # Use the distribution of normally augmented view as supervision label
        labels_ddm = logits.clone().detach()
        labels_ddm = torch.softmax(labels_ddm, dim=1)
        labels_ddm = labels_ddm.detach()
        
        labels_ddm2 = logits_1_ddm.clone().detach()
        labels_ddm2 = torch.softmax(labels_ddm2, dim=1)
        labels_ddm2 = labels_ddm2.detach()

        # labels_ddm_tem = logits_tem_ddm.clone().detach()
        # labels_ddm_tem = torch.softmax(labels_ddm_tem, dim=1)
        # labels_ddm_tem = labels_ddm_tem.detach()
        # dequeue and enqueue
        self._dequeue_and_enqueue(k)
        # tsne feature
        # if tsne_features:
        #     with torch.no_grad():
        #         # self.tsne_pn_monitor([torch.softmax(logits, dim=1), logits_1, torch.softmax(logits_tem,dim=1)], epoch)
        #         np.save(f'./save_tensor/ntu60_xsub_motion_45gated_q1.npy',torch.cat(self.q_tem_feature_bank, dim=0).cpu().detach().numpy())
        #         np.save(f'./save_tensor/ntu60_xsub_motion_45gated_kque.npy',self.queue.cpu().detach().numpy())
        #         self.tsne_pn_monitor([torch.cat(self.q_tem_feature_bank, dim=0).contiguous()[3*256:4*256, :], self.queue[:, 2*256:3*256].T], epoch)
        #
        #         # self.tsne_pn_monitor([q_tem[:128, :], self.queue[:, :128].T], epoch)
        # else:
        #     self.q_tem_feature_bank.append(q_1)
        # distribution
        # p_pos = torch.squeeze(torch.exp(l_pos))/(torch.squeeze(torch.exp(l_pos))+torch.sum(torch.squeeze(torch.exp(l_neg)), dim=1))
        # p1_pos = torch.squeeze(torch.exp(l_pos_t))/(torch.squeeze(torch.exp(l_pos_t))+torch.sum(torch.squeeze(torch.exp(l_neg_t)), dim=1))
        # return logits, labels, logits_1, logits_2, logits_tem, labels_ddm, labels_ddm2,[p_pos,p1_pos]
        return logits, labels, logits_1, logits_2, logits_tem, labels_ddm, labels_ddm2
    @torch.no_grad()
    def tsne_pn_monitor(self, embeding_list, epoch):
        tsne = TSNE(n_components=2, n_iter=1000, init='pca', random_state=0)
        feature_bank = torch.cat(embeding_list,dim=0).contiguous()
        label_bank = torch.tensor([0]*256+[1]*256).to(feature_bank.device)
        tsne_results = tsne.fit_transform(feature_bank.detach().cpu().numpy())
        scatter=plt.scatter(tsne_results[:, 0], tsne_results[:, 1], s=15, c=label_bank.cpu().numpy(), cmap='Paired', alpha=0.75)
        plt.legend(handles=scatter.legend_elements()[0],labels=['Strong positive', 'Normal negative'], title="Views",loc='upper left',prop = {'size':8})
        plt.savefig(
            f'/disk/XhWorks/h2/skeleton_models/HiCLR-main/tsne_visualization/45gated/positive_negative_ns/ntu60_ep{epoch}_xsub_joint_wweights_qtemn_45gated.png',
            dpi=1000, bbox_inches='tight')
        print('tsne done')
        # 3d
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # # ax.scatter(tsne_results[:, 0], tsne_results[:, 1],c=label_bank.cpu().numpy(), cmap='viridis')
        # ax.legend()
        # # ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:,2], cmap='viridis')
        # ax.scatter(tsne_results[:, 0], tsne_results[:, 1], tsne_results[:, 2], s=15, c=label_bank.cpu().numpy(),
        #            cmap='tab20')
        # # fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10), subplot_kw={'projection': '3d'})
        # plt.savefig(
        #     f'/disk/XhWorks/h2/skeleton_models/HiCLR-main/tsne_visualization/45gated/positive_negative_ns/test_45gated_3d_{epoch}_kqtemn.png')
    def nearest_neighbors_mining_mutalddm_wmask(self, im_q, im_k, im_q_1, im_q_2, im_q_tem, topk=1, mask=None, im_k_str=None):
        '''
        The Nearest Neighbors Mining in AimCLR
        '''

        q = self.encoder_q(im_q)  # NxC        
        q_1 = self.encoder_q(im_q_1, drop_graph=False)  # NxC
        q_2 = self.encoder_q(im_q_2, drop_graph=False)  # NxC
        q_tem = self.encoder_q(im_q_tem, drop_graph=False) #N*C
        # Normalize the feature
        q = F.normalize(q, dim=1)
        q_1 = F.normalize(q_1, dim=1)
        q_2 = F.normalize(q_2, dim=1)
        q_tem = F.normalize(q_tem, dim=1)

        # Compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)  # keys: NxC
            k = F.normalize(k, dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        l_pos_e = torch.einsum('nc,nc->n', [q_1, k]).unsqueeze(-1)
        l_neg_e = torch.einsum('nc,ck->nk', [q_1, self.queue.clone().detach()])

        l_pos_ed = torch.einsum('nc,nc->n', [q_2, k]).unsqueeze(-1)
        l_neg_ed = torch.einsum('nc,ck->nk', [q_2, self.queue.clone().detach()])

        l_pos_tem = torch.einsum('nc,nc->n', [q_tem, k]).unsqueeze(-1)
        l_neg_tem = torch.einsum('nc,ck->nk', [q_tem, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits_1 = torch.cat([l_pos_e, l_neg_e], dim=1) # n*(k+1)
        logits_2 = torch.cat([l_pos_ed, l_neg_ed], dim=1)
        logits_tem = torch.cat([l_pos_tem, l_neg_tem], dim=1)

        logits /= self.T
        logits_1 /= self.T
        logits_1_ddm = logits_1.clone()

        logits_2 /= self.T
        logits_tem /= self.T
        # logits_tem_ddm = logits_tem.clone()

        logits_1 = torch.softmax(logits_1, dim=1)
        logits_2 = torch.softmax(logits_2, dim=1)
        # logits_tem = torch.softmax(logits_tem, dim=1)
        # Use the distribution of normally augmented view as supervision label
        labels_ddm = logits.clone().detach()
        labels_ddm = torch.softmax(labels_ddm, dim=1)
        labels_ddm = labels_ddm.detach()

        labels_ddm2 = logits_1_ddm.clone().detach()
        labels_ddm2 = torch.softmax(labels_ddm2, dim=1)
        labels_ddm2 = labels_ddm2.detach()

        # labels_ddm_tem = logits_tem_ddm.clone().detach()
        # labels_ddm_tem = torch.softmax(labels_ddm_tem, dim=1)
        # labels_ddm_tem = labels_ddm_tem.detach()
        # nearest neighbors mining to expand the positive set
        _, topkdix = torch.topk(l_neg, topk, dim=1)
        _, topkdix_e = torch.topk(l_neg_e, topk, dim=1)
        _, topkdix_ed = torch.topk(l_neg_ed, topk, dim=1)
        _, topkdix_tem = torch.topk(l_neg_tem, topk, dim=1)

        topk_onehot = torch.zeros_like(l_neg)

        topk_onehot.scatter_(1, topkdix, 1)
        topk_onehot.scatter_(1, topkdix_e, 1)
        topk_onehot.scatter_(1, topkdix_ed, 1)
        topk_onehot.scatter_(1, topkdix_tem, 1)

        pos_mask = torch.cat([torch.ones(topk_onehot.size(0), 1).cuda(), topk_onehot], dim=1)

        self._dequeue_and_enqueue(k)

        return logits, pos_mask, logits_1, logits_2, logits_tem, labels_ddm, labels_ddm2