from Models.BasicGNNs import *
from Models.FraGAT.ClassifierModel import DNN
from Models.Graphormer.Graphormer import EncoderLayer, init_params
import torch as t
import torch.nn.functional as F
from TrainingFramework.ProcessControllers import Configs

class ProjectHead(nn.Module):
    def __init__(self, opt):
        super(ProjectHead, self).__init__()
        self.opt = opt

        project_head_size = self.opt.args['ProjectHeadSize']
        FP_size = self.opt.args['FPSize']

        self.LayerList = nn.ModuleList()

        self.LayerList.append(nn.Linear(FP_size, project_head_size))
        self.LayerList.append(nn.BatchNorm1d(project_head_size, eps=1e-06, momentum=0.1))
        self.LayerList.append(nn.ReLU(inplace=True))
        self.LayerList.append(nn.Linear(project_head_size, project_head_size//2))
        self.Drop = nn.Dropout(p=self.opt.args['DropRate'])

    def forward(self,x):
        for layer in self.LayerList:
            x = layer(x)
        x = self.Drop(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, opt):
        super(TransformerEncoder, self).__init__()
        self.opt = opt
        FPSize = self.opt.args['FPSize']
        hidden_size = self.opt.args['TransformerHiddenSize']
        ffn_size = self.opt.args['TransformerFFNSize']
        drop_rate = self.opt.args['DropRate']
        attention_drop_rate = self.opt.args['DropRate']
        num_heads = self.opt.args['TransformerNumHeads']
        num_layers = self.opt.args['TransformerEncoderLayers']

        self.input_ln = nn.Linear(FPSize, hidden_size)
        self.output_ln = nn.Linear(hidden_size, FPSize)

        encoders = [EncoderLayer(hidden_size, ffn_size, drop_rate, attention_drop_rate, num_heads)
                    for _ in range(num_layers)]
        self.encoders = nn.ModuleList(encoders)

        self.apply(lambda module: init_params(module, n_layers=num_layers))


    def forward(self,Input):
        hidden = self.input_ln(Input)
        for enc_layer in self.encoders:
            hidden = enc_layer(hidden)
        output = self.output_ln(hidden)
        return output


class FraSICL(nn.Module):
    def __init__(self, opt):
        super(FraSICL, self).__init__()
        self.opt = opt

        self.MolGraphBackbone = self.MakeBackboneModel('mol')
        self.FragGraphBakebone = self.MakeBackboneModel('frag')
        self.MolProj = ProjectHead(opt)
        self.FragProj = ProjectHead(opt)
        self.FragViewProj = ProjectHead(opt)
        self.FragTransEncoder = TransformerEncoder(opt)

        self.device = t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')



    def forward(self, Input, finetune=False, usewhichview = None):
        Input_frag, Input_mol, Input_SMILES = Input
        # print(f"Input_mol:{Input_mol}")
        # print(f"Input_frag:{Input_frag}")
        if not finetune:
            Input_mol = Input_mol.to(self.device)
            Input_frag = Input_frag.to(self.device)
        # print(f"Input_mol:{Input_mol.x}")
        # raise RuntimeError
        # print(f"self.MolGraphBackbone.device:{next(self.MolGraphBackbone.parameters()).device}")
        MolEmbeddings = self.MolGraphBackbone(Input_mol)
        # print(f"MolEmbeddings:{MolEmbeddings}")
        if finetune:
            if usewhichview == 'Mol':
                return MolEmbeddings                            # MolEmbeddings: [BatchSize, FPSize]
        MolEmbeddingsProj = self.MolProj(MolEmbeddings)         # MolEmbeddings(proj): [BatchSize, ProjSize/2]
        MolEmbeddingsProj = F.normalize(MolEmbeddingsProj, dim=1)

        #####################
        batch_size = Input_frag.singlebond_num.size()[0]

        cumulated_graph_nums = self.CalCumulatedGraphNum(Input_frag)

        Input_frag.batch = self.FragBatchTransform(Input_frag, batch_size, cumulated_graph_nums)

        FragEmbeddings = self.FragGraphBakebone(Input_frag)    # FragEmbeddings: [sum(fraggraphnums), FPSize]
        # sum(fraggraphnums) = 2 * sum(singlebond_nums) = 2 * sum(fragpairnums)
        if not finetune:
            fragpairnums = sum(Input_frag.singlebond_num)
        else:
            singlebond_num = Input_frag.singlebond_num.clone()
            singlebond_num[singlebond_num==0]=1
            fragpairnums = sum(singlebond_num)


        if finetune:
            FragEmbeddings = self.FragEmbeddings_view(FragEmbeddings, batch_size, cumulated_graph_nums, Input_frag.singlebond_num)
            # print(f"FragEmbeddings.size():{FragEmbeddings.size()}")
            # print(f"fragpairnums:{fragpairnums}")
            assert FragEmbeddings.size()[0] == fragpairnums
            # raise RuntimeError
        else:
            # print(f"cumulated_graph_nums:{cumulated_graph_nums}")
            # print(f"Input_frag.singlebond_num:{Input_frag.singlebond_num}")

            assert False not in (FragEmbeddings.view(fragpairnums,2,-1)[0,1,:] == FragEmbeddings[1,:])
            assert False not in (FragEmbeddings.view(fragpairnums,2,-1)[1,1,:] == FragEmbeddings[3,:])



            FragEmbeddings = FragEmbeddings.view(fragpairnums, 2, -1)
            FragEmbeddings = FragEmbeddings.sum(dim=1)             # FragEmbeddings: [sum(fragpairnums), FPSize]

        # print(f"FragEmbeddings:{FragEmbeddings}")
        # print(f"FragEmbeddings.size():{FragEmbeddings.size()}")
        # raise RuntimeError
        FragEmbeddingsProj = self.FragProj(FragEmbeddings)         # FragEmbeddings(proj): [sum(fragpairnums), ProjSize/2]
        FragEmbeddingsProj = F.normalize(FragEmbeddingsProj, dim=1)   # FragEmbeddings(proj): [sum(fragpairnums), ProjSize/2]
        FragEmbedsSim = self._cal_dot_similarity(FragEmbeddingsProj, FragEmbeddingsProj)
        # FragEmbedsSim: [sum(fragpairnums), sum(fragpairnums)

        ######################
        # For transformer, only the relationship between the views of each mol should be calculated
        # generate a batched data of FragEmbeddings
        # For each mol, the number of views is singlebond_num[i]
        BatchedFragEmbeddings = self.BuildBatchFragEmbeddingsForTransformer(FragEmbeddings, Input_frag.singlebond_num)
        # BatchedFragEmbeddings: [BatchSize, MaxFragPairNum(pad), FPSize]
        BatchedFragEmbeddings = self.FragTransEncoder(BatchedFragEmbeddings)
        # BatchedFragEmbeddings: [BatchSize, MaxFragPairNum(pad), FPSize]

        FragViewEmbeddings = self.ReadoutFragEmbeddings(BatchedFragEmbeddings,Input_frag.singlebond_num)
        if finetune:
            if usewhichview == 'Frag':
                return FragViewEmbeddings
        # FragViewEmbeddings: [BatchSize, FPSize]
        FragViewEmbeddingsProj = self.FragViewProj(FragViewEmbeddings)
        FragViewEmbeddingsProj = F.normalize(FragViewEmbeddingsProj,dim=1)
        # FragViewEmbeddingsProj: [BatchSize, ProjSize/2]


        return MolEmbeddingsProj, FragViewEmbeddingsProj, FragEmbedsSim

    def FragEmbeddings_view(self, FragEmbeddings, batch_size, cumulated_graph_nums, singlebond_num):
        viewed_FragEmbeddings = []
        start_idx = 0
        for i in range(batch_size):
            if singlebond_num[i] != 0:
                for j in range(singlebond_num[i]):
                    viewed_FragEmbeddings.append(FragEmbeddings[start_idx:(start_idx+2),:].sum(dim=0).unsqueeze(0))
                    start_idx += 2
            else:
                viewed_FragEmbeddings.append(FragEmbeddings[start_idx:(start_idx+1),:])
                start_idx += 1

        return t.cat(viewed_FragEmbeddings,dim=0)


    def _cal_dot_similarity(self, x,y):
        v = t.tensordot(x.unsqueeze(1),y.T.unsqueeze(0),dims=2)
        return v

    def ReadoutFragEmbeddings(self, FragEmbeddings, singlebond_nums):
        # singlebond_num = singlebond_num
        BatchedFragViewEmbeddings = []
        for i in range(len(singlebond_nums)):
            if singlebond_nums[i] == 0:
                singlebond_num = 1
            else:
                singlebond_num = singlebond_nums[i]
            cur_mol_frag_view_embedding = FragEmbeddings[i,0:singlebond_num,:].mean(dim=0)
            BatchedFragViewEmbeddings.append(cur_mol_frag_view_embedding.unsqueeze(0))
        BatchedFragViewEmbeddings = t.cat(BatchedFragViewEmbeddings, dim=0)
        # print(f"Readout of fragment view embeddings: {BatchedFragViewEmbeddings.size()}")
        return BatchedFragViewEmbeddings


    def BuildBatchFragEmbeddingsForTransformer(self, FragEmbeddings, singlebond_nums):
        start_idx = 0
        max_embedding_num = singlebond_nums.max()
        FragEmbeddingList = []
        for i in range(len(singlebond_nums)):
            if singlebond_nums[i] == 0:
                singlebond_num = 1
            else:
                singlebond_num = singlebond_nums[i]

            cur_mol_embeddings = FragEmbeddings[start_idx:(start_idx+singlebond_num),:]
            # print(f"cur_mol_embeddings:{cur_mol_embeddings}")
            assert cur_mol_embeddings.size()[0] == singlebond_num
            cur_mol_embeddings_pad = nn.functional.pad(cur_mol_embeddings, (0,0,0,max_embedding_num-cur_mol_embeddings.size()[0]), mode='constant', value=0)
            FragEmbeddingList.append(cur_mol_embeddings_pad.unsqueeze(0))
            start_idx += singlebond_num
        FragEmbeddings = t.cat(FragEmbeddingList,dim=0)
        # print(f"singlebond_nums:{singlebond_nums}")
        # print(f"FragmentEmbeddings.size():{FragEmbeddings.size()}")
        # raise RuntimeError
        return FragEmbeddings


    def CalCumulatedGraphNum(self, Input):
        # todo(zqzhagn): updated 9.3
        # singlebond_num = Input.singlebond_num
        singlebond_num = Input.singlebond_num.clone()
        singlebond_num = singlebond_num * 2
        singlebond_num[singlebond_num==0] = 1     # only one graph when singlbebond_num == 0, i.e. the OriMol, and no pad now.
        cumulated_graph_nums = singlebond_num.cumsum(dim=0)
        # print(f"singlebond_num:{Input.singlebond_num}")
        # print(f"graph_num:{singlebond_num}")
        # print(f"cumulated_graph_nums:{cumulated_graph_nums}")
        return cumulated_graph_nums

    def FragBatchTransform(self, Input, batch_size, cumulated_graph_nums):
        mask_frags_all = Input.mask_frags[Input.mask_frags!=-1]
        batch = Input.batch.clone()
        for i in range(1,batch_size):
            batch[Input.batch==i] = cumulated_graph_nums[i-1]

        batch = mask_frags_all + batch
        return batch


    def MakeBackboneModel(self,mode):
        if mode == 'mol':
            backbone_choice = self.opt.args['BackbonePretrainModel-mol']
        elif mode == 'frag':
            backbone_choice = self.opt.args['BackbonePretrainModel-frag']
        else:
            raise KeyError

        if backbone_choice == 'PyGGINE':
            net = PyGGIN(self.opt, FeatureExtractor = True)
        elif backbone_choice == 'PyGATFP':
            if mode == 'mol':
                args_mol = {'AtomFeatureSize':self.opt.args['AtomFeatureSize'],
                            'BondFeatureSize':self.opt.args['BondFeatureSize'],
                            'ATFPInputSize':self.opt.args['ATFPInputSize_mol'],
                            'ATFPHiddenSize':self.opt.args['ATFPHiddenSize_mol'],
                            'FPSize':self.opt.args['FPSize'],
                            'AtomLayers':self.opt.args['AtomLayers_mol'],
                            'MolLayers':self.opt.args['MolLayers_mol'],
                            'DropRate':self.opt.args['DropRate'],
                            'CUDA_VISIBLE_DEVICES':self.opt.args['CUDA_VISIBLE_DEVICES']}
                opt_mol = Configs(args_mol)
                net = PyGATFP(opt_mol,FeatureExtractor = True)
            elif mode == 'frag':
                args_frag = {'AtomFeatureSize': self.opt.args['AtomFeatureSize'],
                            'BondFeatureSize': self.opt.args['BondFeatureSize'],
                            'ATFPInputSize': self.opt.args['ATFPInputSize_frag'],
                            'ATFPHiddenSize': self.opt.args['ATFPHiddenSize_frag'],
                            'FPSize': self.opt.args['FPSize'],
                            'AtomLayers': self.opt.args['AtomLayers_frag'],
                            'MolLayers': self.opt.args['MolLayers_frag'],
                            'DropRate': self.opt.args['DropRate'],
                            'CUDA_VISIBLE_DEVICES': self.opt.args['CUDA_VISIBLE_DEVICES']}
                opt_frag = Configs(args_frag)
                net = PyGATFP(opt_frag, FeatureExtractor = True)
        else:
            raise NotImplementedError

        return net
