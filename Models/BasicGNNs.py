import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCN, global_add_pool, global_mean_pool, global_max_pool, MLP, GIN, SGConv, MessagePassing, AttentiveFP, GraphSAGE, GINEConv
from torch_geometric.nn.models.basic_gnn import BasicGNN



class PyGGCN(nn.Module):
    def __init__(self, opt, FeatureExtractor = False):
        super(PyGGCN, self).__init__()
        self.opt = opt
        self.node_feat_size = opt.args['AtomFeatureSize']
        self.in_channel = opt.args['GCNInputSize']
        self.hidden_channel = opt.args['GCNHiddenSize']
        self.out_channel = opt.args['FPSize']
        self.num_layers = opt.args['GCNLayers']
        self.MLPChannels = opt.args['DNNLayers']
        self.MLPOutputSize = opt.args['OutputSize']
        self.dropout = opt.args['DropRate']
        self.FeatureExtractor = FeatureExtractor

        self.MLPChannels = [self.out_channel] + self.MLPChannels + [self.MLPOutputSize]

        self.GCN = GCN(in_channels = self.in_channel,
                       hidden_channels = self.hidden_channel,
                       out_channels = self.out_channel,
                       num_layers = self.num_layers,
                       dropout = self.dropout)
        self.NodeFeatEmbed = MLP([self.node_feat_size, self.in_channel], dropout = self.dropout)
        if not self.FeatureExtractor:
            self.TaskLayer = MLP(self.MLPChannels, dropout = self.dropout)

        self.ReadoutList = {
            'Add': global_add_pool,
            'Mean': global_mean_pool,
            'Max': global_max_pool
        }
        self.readout = self.ReadoutList[opt.args['GCNReadout']]
        self.device = t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')

    def forward(self, Input):
        # Input: Batch data of PyG
        Input = Input.to(self.device)
        x = self.NodeFeatEmbed(Input.x)
        x = self.GCN(x, Input.edge_index)
        x = self.readout(x, Input.batch)
        if not self.FeatureExtractor:
            x = self.TaskLayer(x)
        return x

class PyGGIN(nn.Module):
    def __init__(self, opt, FeatureExtractor = False):
        super(PyGGIN, self).__init__()
        self.opt = opt
        self.node_feat_size = opt.args['AtomFeatureSize']
        self.in_channel = opt.args['GINInputSize']
        self.hidden_channel = opt.args['GINHiddenSize']
        self.out_channel = opt.args['FPSize']
        self.eps = opt.args['GINEps']
        self.num_layers = opt.args['GINLayers']
        self.MLPChannels = opt.args['DNNLayers']
        self.MLPOutputSize = opt.args['OutputSize']
        self.dropout = opt.args['DropRate']
        self.FeatureExtractor = FeatureExtractor

        self.MLPChannels = [self.out_channel] + self.MLPChannels + [self.MLPOutputSize]

        self.GIN = GIN(in_channels = self.in_channel,
                       hidden_channels = self.hidden_channel,
                       out_channels = self.out_channel,
                       num_layers = self.num_layers,
                       dropout = self.dropout,
                       eps = self.eps)
        self.NodeFeatEmbed = MLP([self.node_feat_size, self.in_channel], dropout = self.dropout)
        if not self.FeatureExtractor:
            self.TaskLayer = MLP(self.MLPChannels, dropout = self.dropout)

        self.ReadoutList = {
            'Add': global_add_pool,
            'Mean': global_mean_pool,
            'Max': global_max_pool,
        }
        self.readout = self.ReadoutList[opt.args['GINReadout']]

    def forward(self, Input):
        # Input: Batch data of PyG
        Input = Input.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
        x = self.NodeFeatEmbed(Input.x)
        x = self.GIN(x, Input.edge_index)
        x = self.readout(x, Input.batch)
        if not self.FeatureExtractor:
            x = self.TaskLayer(x)
        return x

class GINE(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int, train_eps=True):
        super(GINE, self).__init__()
        self.train_eps = train_eps
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.atom_embed = nn.Linear(in_channels, hidden_channels)
        self.bond_embed = nn.Linear(in_channels, hidden_channels)
        for _ in range(num_layers):
            layer = nn.Sequential(
                nn.Linear(hidden_channels, 2 * hidden_channels),
                nn.BatchNorm1d(2 * hidden_channels),
                nn.ReLU(),
                nn.Linear(2 * hidden_channels, hidden_channels)
            )
            self.convs.append(GINEConv(layer, self.train_eps))
            self.norms.append(nn.BatchNorm1d(hidden_channels))
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = self.atom_embed(x)
        edge_attr = self.bond_embed(edge_attr)
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = self.norms[i](x)
            x = F.relu(x)
        if self.num_layers > 1:
            x = self.lin(x)
        return x

class GINet(nn.Module):
    """
    in_channels: dimensionality of atom_embeddings, bond_embeddings
    out_channels: dimensionality of graph_embeddings
    """
    def __init__(self, num_layers, in_channels, hidden_channels,
                 out_channels, train_eps=True, pool='add'):
        super(GINet, self).__init__()
        self.num_layers = num_layers
        self.in_channel = in_channels
        self.hidden_channel = hidden_channels
        self.out_channel = out_channels
        self.train_eps = train_eps
        self.node_feat_size = opt.args['AtomFeatureSize']


        self.GINE = GINE(in_channels = self.in_channel,
                         hidden_channels = self.hidden_channel,
                         out_channels = self.out_channel,
                         num_layers = self.num_layers,
                         train_eps = self.train_eps)

        self.NodeFeatEmbed = MLP([self.node_feat_size, self.in_channel], dropout = self.dropout)
        self.BondFeatEmbed = MLP([self.bond_feat_size, self.in_channel], dropout = self.dropout)
        if pool == 'mean':
            self.pool = global_max_pool
        elif pool == 'max':
            self.pool = global_max_pool
        elif pool == 'add':
            self.pool = global_add_pool

    def forward(self, data):
        x = data.x
        edge_attr = data.edge_attr
        edge_index = data.edge_index
        x = self.NodeFeatEmbed(x)
        edge_attr = self.BondFeatEmbed(edge_attr)
        x = self.GINE(x, edge_index, edge_attr)
        x = self.pool(x, data.batch)

        return x


class SGC(BasicGNN):
    def init_conv(self, in_channels: int, out_channels: int, **kwargs) -> MessagePassing:
        return SGConv(in_channels, out_channels, **kwargs)

class PyGSGC(nn.Module):
    def __init__(self, opt, FeatureExtractor = False):
        super(PyGSGC, self).__init__()
        self.opt = opt
        self.node_feat_size = opt.args['AtomFeatureSize']
        self.in_channel = opt.args['SGCInputSize']
        self.hidden_channel = opt.args['SGCHiddenSize']
        self.out_channel = opt.args['FPSize']
        self.K = opt.args['SGCK']
        self.num_layers = opt.args['SGCLayers']
        self.MLPChannels = opt.args['DNNLayers']
        self.MLPOutputSize = opt.args['OutputSize']
        self.dropout = opt.args['DropRate']
        self.FeatureExtractor = FeatureExtractor


        self.MLPChannels = [self.out_channel] + self.MLPChannels + [self.MLPOutputSize]

        self.SGC = SGC(in_channels = self.in_channel,
                       hidden_channels = self.hidden_channel,
                       out_channels = self.out_channel,
                       num_layers = self.num_layers,
                       dropout = self.dropout,
                       K = self.K)
        self.NodeFeatEmbed = MLP([self.node_feat_size, self.in_channel], dropout = self.dropout)
        self.TaskLayer = MLP(self.MLPChannels, dropout = self.dropout)

        self.ReadoutList = {
            'Add': global_add_pool,
            'Mean': global_mean_pool,
            'Max': global_max_pool
        }
        self.readout = self.ReadoutList[opt.args['SGCReadout']]

    def forward(self, Input):
        # Input: Batch data of PyG
        Input = Input.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
        x = self.NodeFeatEmbed(Input.x)
        x = self.SGC(x, Input.edge_index)
        x = self.readout(x, Input.batch)
        if not self.FeatureExtractor:
            x = self.TaskLayer(x)
        return x


# todo(zqzhang): updated in TPv8
class PyGATFP(nn.Module):
    def __init__(self, opt, FeatureExtractor = False, JT=False):
        super(PyGATFP, self).__init__()
        self.opt = opt
        self.FeatureExtractor = FeatureExtractor

        if JT:
            self.node_feat_size = opt.args['FPSize']
        else:
            self.node_feat_size = opt.args['AtomFeatureSize']
        self.edge_feat_size = opt.args['BondFeatureSize']
        self.in_channel = opt.args['ATFPInputSize']
        self.hidden_channel = opt.args['ATFPHiddenSize']
        self.out_channel = opt.args['FPSize']
        self.edge_dim = self.edge_feat_size
        self.num_layers = opt.args['AtomLayers']
        self.num_timesteps = opt.args['MolLayers']
        self.dropout = opt.args['DropRate']

        self.NodeFeatEmbed = MLP([self.node_feat_size, self.in_channel], dropout = self.dropout)

        self.ATFP = AttentiveFP(in_channels = self.in_channel,
                                hidden_channels = self.hidden_channel,
                                out_channels = self.out_channel,
                                edge_dim = self.edge_dim,
                                num_layers = self.num_layers,
                                num_timesteps = self.num_timesteps,
                                dropout = self.dropout)

        if not self.FeatureExtractor:
            self.OutputSize = opt.args['OutputSize']
            self.MLPChannels = opt.args['DNNLayers']
            self.MLPChannels = [self.out_channel] + self.MLPChannels + [self.OutputSize]
            self.TaskLayer = MLP(self.MLPChannels, dropout = self.dropout)

        self.device = t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')

    def forward(self, Input):
        if Input.x.device != next(self.NodeFeatEmbed.parameters()).device:
            Input = Input.to(self.device)

        x = self.NodeFeatEmbed(Input.x)
        x = self.ATFP(x, Input.edge_index, Input.edge_attr, Input.batch)
        if not self.FeatureExtractor:
            x = self.TaskLayer(x)
        return x

class PyGGraphSAGE(nn.Module):
    def __init__(self, opt, FeatureExtractor = False):
        super(PyGGraphSAGE, self).__init__()
        self.opt = opt
        self.FeatureExtractor = FeatureExtractor

        self.node_feat_size = opt.args['AtomFeatureSize']
        self.edge_feat_size = opt.args['BondFeatureSize']
        self.in_channel = opt.args['GraphSAGEInputSize']
        self.hidden_chiannel = opt.args['GraphSAGEHiddenSize']
        self.out_channel = opt.args['FPSize']
        self.dropout = opt.args['DropRate']
        # self.act = opt.args['GraphSAGEAct']
        self.num_layers = opt.args['GraphSAGELayers']
        self.MLPChannels = opt.args['DNNLayers']
        self.MLPOutputSize = opt.args['OutputSize']

        self.MLPChannels = [self.out_channel] + self.MLPChannels + [self.MLPOutputSize]

        self.GraphSAGE = GraphSAGE(in_channels = self.in_channel,
                                   hidden_channels = self.hidden_chiannel,
                                   num_layers = self.num_layers,
                                   out_channels = self.out_channel,
                                   dropout = self.dropout)
                                   # act = self.act)

        self.NodeFeatEmbed = MLP([self.node_feat_size, self.in_channel], dropout = self.dropout)
        if not self.FeatureExtractor:
            self.TaskLayer = MLP(self.MLPChannels, dropout = self.dropout)

        self.ReadoutList = {
            'Add': global_add_pool,
            'Mean': global_mean_pool,
            'Max': global_max_pool
        }
        self.readout = self.ReadoutList[opt.args['GraphSAGEReadout']]
        self.device = t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')

    def forward(self, Input):
        Input = Input.to(self.device)
        x = self.NodeFeatEmbed(Input.x)
        x = self.GraphSAGE(x, Input.edge_index)
        x = self.readout(x, Input.batch)
        if not self.FeatureExtractor:
            x = self.TaskLayer(x)
        return x