import torch as t
import torch.nn as nn
import numpy as np

class NTXentLoss(t.nn.Module):
    def __init__(self,opt):
        super(NTXentLoss, self).__init__()
        self.opt = opt
        self.batch_size = self.opt.args['BatchSize']
        self.temperature = self.opt.args['NTXentLossTemp']
        self.device = t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')
        self.softmax = nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(t.bool)
        self.criterion = nn.CrossEntropyLoss(reduction = 'sum')

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k = -self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k = self.batch_size)
        mask = t.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(t.bool)
        return mask.to(self.device)

    def forward(self, MolEmbeddings, FragViewEmbeddings):
        Embeddings = t.cat([MolEmbeddings, FragViewEmbeddings], dim=0)
        similarity_mat = self._cal_dot_similarity(Embeddings, Embeddings)
        # [ batch_size + batch_size, FPSize]
        # find positive samples:
        l_pos = t.diag(similarity_mat, self.batch_size)
        r_pos = t.diag(similarity_mat, -self.batch_size)
        positives = t.cat([l_pos, r_pos]).view(2*self.batch_size, 1)

        negatives = similarity_mat[self.mask_samples_from_same_repr].view(2*self.batch_size, -1)

        logits = t.cat([positives, negatives],dim = 1)
        logits /= self.temperature

        labels = t.zeros(2*self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss/ (2*self.batch_size)


    def _cal_dot_similarity(self, x,y):
        v = t.tensordot(x.unsqueeze(1),y.T.unsqueeze(0),dims=2)
        return v