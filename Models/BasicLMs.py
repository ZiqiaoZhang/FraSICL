import torch as t
import torch.nn as nn
from Models.FraGAT.FraGATModel import DNN


class MolPredLSTM(nn.Module):
    def __init__(self, opt):
        super(MolPredLSTM, self).__init__()
        self.opt = opt
        self.WordEmbed = nn.Embedding(self.opt.args['MaxDictLength'],
                                      self.opt.args['FPSize'],
                                      padding_idx = self.opt.args['MaxDictLength']-1)
        self.MolFeatureExtractor = nn.LSTM(input_size = self.opt.args['FPSize'],
                                           hidden_size = self.opt.args['FPSize'],
                                           num_layers = self.opt.args['LSTMLayers'],
                                           batch_first = True,
                                           bidirectional = True)
        self.Classifier = DNN(input_size=self.opt.args['FPSize'],
                              output_size = self.opt.args['OutputSize'],
                              layer_sizes = self.opt.args['DNNLayers'],
                              opt = self.opt)
        self.device = t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')
    def forward(self, Input):
        Input = Input.to(self.device)
        Embed = self.WordEmbed(Input)
        _, (MolFeature, _) = self.MolFeatureExtractor(Embed)
        # MolFeature: [LSTMLayer * Bi, batchsize, FP_Size]
        MolFeature = MolFeature.permute(1,0,2)
        # MolFeature: [batchsize, LSTMLayer * Bi, FP_Size]
        MolFeature = MolFeature.sum(dim=1)
        # MolFeature: [batchsie, FPSize]
        prediction = self.Classifier(MolFeature)
        return prediction

class MolPredGRU(nn.Module):
    def __init__(self, opt):
        super(MolPredGRU, self).__init__()
        self.opt = opt
        self.WordEmbed = nn.Embedding(self.opt.args['MaxDictLength'],
                                      self.opt.args['FPSize'],
                                      padding_idx = self.opt.args['MaxDictLength'] - 1)
        self.MolFeatureExtractor = nn.GRU(input_size = self.opt.args['FPSize'],
                                           hidden_size = self.opt.args['FPSize'],
                                           num_layers = self.opt.args['GRULayers'],
                                           batch_first = True,
                                           bidirectional = True)
        self.Classifier = DNN(
                input_size = self.opt.args['FPSize'],
                output_size = self.opt.args['OutputSize'],
                layer_sizes = self.opt.args['DNNLayers'],
                opt = self.opt)
        self.device = t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')

    def forward(self, Input):
        Input = Input.to(self.device)
        Embed = self.WordEmbed(Input)
        _, MolFeature = self.MolFeatureExtractor(Embed)
        MolFeature = MolFeature.permute(1,0,2)
        MolFeature = MolFeature.sum(dim=1)
        prediction = self.Classifier(MolFeature)
        return prediction

