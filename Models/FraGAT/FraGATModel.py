from Models.FraGAT.ClassifierModel import DNN
from Models.FraGAT.AttentiveFP import *
from Models.BasicGNNs import *
from torch_geometric.data import Data
import random
import numpy as np
import time


class MolPredFraGAT(nn.Module):
    # Model of FraGAT
    # including three branches: original mol_graph, Frags and JT.
    # Four latent vectors are extracted from these three branches and concated
    def __init__(self,
                 atom_feature_size,
                 bond_feature_size,
                 FP_size,
                 atom_layers,
                 mol_layers,
                 DNN_layers,
                 output_size,
                 droprate,
                 opt):
        super(MolPredFraGAT, self).__init__()
        self.AtomEmbedding = AttentiveFP_atom(
            atom_feature_size=atom_feature_size,
            bond_feature_size=bond_feature_size,
            FP_size=FP_size,
            layers=atom_layers,
            droprate=droprate
        )    # Branch for Frags and original mol_graph
        self.MolEmbedding = AttentiveFP_mol(
            layers=mol_layers,
            FP_size=FP_size,
            droprate=droprate
        )    # Mol Embedding Module can be used repeatedly
        self.Classifier = DNN(
            input_size=4*FP_size,
            output_size=output_size,
            layer_sizes=DNN_layers,
            opt=opt
        )
        self.AtomEmbeddingJT = AttentiveFP_atom(
            atom_feature_size=FP_size,
            bond_feature_size=bond_feature_size,
            FP_size=FP_size,
            layers=atom_layers,
            droprate=droprate
        )    # Branch for Junction Tree.
        self.opt = opt

    def forward(self, Input):
        [atom_features,
         bond_features,
         atom_neighbor_list_origin,
         bond_neighbor_list_origin,
         atom_mask_origin,
         atom_neighbor_list_changed,
         bond_neighbor_list_changed,
         frag_mask1,
         frag_mask2,
         bond_index,
         JT_bond_features,
         JT_atom_neighbor_list,
         JT_bond_neighbor_list,
         JT_mask] = self.Input_cuda(Input)

        # branch origin
        atom_FP_origin = self.AtomEmbedding(atom_features, bond_features, atom_neighbor_list_origin, bond_neighbor_list_origin)
        mol_FP_origin, _ = self.MolEmbedding(atom_FP_origin, atom_mask_origin)

        # branch Frag:
        atom_FP = self.AtomEmbedding(atom_features, bond_features, atom_neighbor_list_changed, bond_neighbor_list_changed)
        mol_FP1, activated_mol_FP1 = self.MolEmbedding(atom_FP, frag_mask1)
        mol_FP2, activated_mol_FP2 = self.MolEmbedding(atom_FP, frag_mask2)
        # mol_FP1, mol_FP2 are used to input the DNN module.
        # activated FPs are used to calculate the mol_FP
        # size: [batch_size, FP_size]
        ###########################################################################################
        # JT construction
        # construct a higher level graph: JT

        # construct atom features of JT:
        # The modules are designed to deal with graphs with pad nodes, so a pad node should be added into the JT graph.
        batch_size, FP_size = activated_mol_FP1.size()
        pad_node_feature = t.zeros(batch_size, FP_size).to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
        JT_atom_features = t.stack([activated_mol_FP1, activated_mol_FP2, pad_node_feature], dim=1)
        # size: [batch_size, 3, FP_size]

        # JT construction complete.
        ###########################################################################################
        # branch JT:
        atom_FP_super = self.AtomEmbeddingJT(JT_atom_features,
                                             JT_bond_features,
                                             JT_atom_neighbor_list,
                                             JT_bond_neighbor_list)
        JT_FP, _ = self.MolEmbedding(atom_FP_super, JT_mask)

        entire_FP = t.cat([mol_FP1, mol_FP2, JT_FP, mol_FP_origin], dim=-1)
        prediction = self.Classifier(entire_FP)
        return prediction, entire_FP

    def Input_cuda(self, Input):
        # to preprocess the input tensors
        [atom_features,
         bond_features,
         atom_neighbor_list_origin,
         bond_neighbor_list_origin,
         atom_mask_origin,
         atom_neighbor_list_changed,
         bond_neighbor_list_changed,
         frag_mask1,
         frag_mask2,
         bond_index,
         JT_bond_features,
         JT_atom_neighbor_list,
         JT_bond_neighbor_list,
         JT_mask] = Input

        # Tensor shapes are different in training and evaluation step
        # the shape of the input tensors is shown in featurizer.py
        if not self.training:
            atom_features = atom_features.squeeze(dim=0).to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            bond_features = bond_features.squeeze(dim=0).to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            atom_neighbor_list_origin = atom_neighbor_list_origin.squeeze(dim=0).to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            bond_neighbor_list_origin = bond_neighbor_list_origin.squeeze(dim=0).to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            atom_mask_origin = atom_mask_origin.squeeze(dim=0).to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))

            atom_neighbor_list_changed = atom_neighbor_list_changed.squeeze(dim=0).to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            bond_neighbor_list_changed = bond_neighbor_list_changed.squeeze(dim=0).to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            frag_mask1 = frag_mask1.squeeze(dim=0).to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            frag_mask2 = frag_mask2.squeeze(dim=0).to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            bond_index = bond_index.squeeze(dim=0).to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))

            JT_bond_features = JT_bond_features.squeeze(dim=0).to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            JT_atom_neighbor_list = JT_atom_neighbor_list.squeeze(dim=0).to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            JT_bond_neighbor_list = JT_bond_neighbor_list.squeeze(dim=0).to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            JT_mask = JT_mask.squeeze(dim=0).to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))

        else:
            atom_features = atom_features.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            bond_features = bond_features.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            atom_neighbor_list_origin = atom_neighbor_list_origin.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            bond_neighbor_list_origin = bond_neighbor_list_origin.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            atom_mask_origin = atom_mask_origin.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))

            atom_neighbor_list_changed = atom_neighbor_list_changed.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            bond_neighbor_list_changed = bond_neighbor_list_changed.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            frag_mask1 = frag_mask1.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            frag_mask2 = frag_mask2.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            bond_index = bond_index.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))

            JT_bond_features = JT_bond_features.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            JT_atom_neighbor_list = JT_atom_neighbor_list.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            JT_bond_neighbor_list = JT_bond_neighbor_list.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            JT_mask = JT_mask.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))

        return [atom_features,
                bond_features,
                atom_neighbor_list_origin,
                bond_neighbor_list_origin,
                atom_mask_origin,
                atom_neighbor_list_changed,
                bond_neighbor_list_changed,
                frag_mask1,
                frag_mask2,
                bond_index,
                JT_bond_features,
                JT_atom_neighbor_list,
                JT_bond_neighbor_list,
                JT_mask]

class MolPredFraGAT_FeatureExtractor(nn.Module):
    # Model of FraGAT
    # including three branches: original mol_graph, Frags and JT.
    # Four latent vectors are extracted from these three branches and concated
    def __init__(self,
                 atom_feature_size,
                 bond_feature_size,
                 FP_size,
                 atom_layers,
                 mol_layers,
                 droprate):
        super(MolPredFraGAT_FeatureExtractor, self).__init__()
        self.AtomEmbedding = AttentiveFP_atom(
            atom_feature_size=atom_feature_size,
            bond_feature_size=bond_feature_size,
            FP_size=FP_size,
            layers=atom_layers,
            droprate=droprate
        )    # Branch for Frags and original mol_graph
        self.MolEmbedding = AttentiveFP_mol(
            layers=mol_layers,
            FP_size=FP_size,
            droprate=droprate
        )    # Mol Embedding Module can be used repeatedly
        # self.Classifier = DNN(
        #     input_size=4*FP_size,
        #     output_size=output_size,
        #     layer_sizes=DNN_layers,
        #     opt=opt
        # )
        self.AtomEmbeddingJT = AttentiveFP_atom(
            atom_feature_size=FP_size,
            bond_feature_size=bond_feature_size,
            FP_size=FP_size,
            layers=atom_layers,
            droprate=droprate
        )    # Branch for Junction Tree.

    def forward(self, Input):
        [atom_features,
         bond_features,
         atom_neighbor_list_origin,
         bond_neighbor_list_origin,
         atom_mask_origin,
         atom_neighbor_list_changed,
         bond_neighbor_list_changed,
         frag_mask1,
         frag_mask2,
         bond_index,
         JT_bond_features,
         JT_atom_neighbor_list,
         JT_bond_neighbor_list,
         JT_mask] = self.Input_cuda(Input)

        # branch origin
        atom_FP_origin = self.AtomEmbedding(atom_features, bond_features, atom_neighbor_list_origin, bond_neighbor_list_origin)
        mol_FP_origin, _ = self.MolEmbedding(atom_FP_origin, atom_mask_origin)

        # branch Frag:
        atom_FP = self.AtomEmbedding(atom_features, bond_features, atom_neighbor_list_changed, bond_neighbor_list_changed)
        mol_FP1, activated_mol_FP1 = self.MolEmbedding(atom_FP, frag_mask1)
        mol_FP2, activated_mol_FP2 = self.MolEmbedding(atom_FP, frag_mask2)
        # mol_FP1, mol_FP2 are used to input the DNN module.
        # activated FPs are used to calculate the mol_FP
        # size: [batch_size, FP_size]
        ###########################################################################################
        # JT construction
        # construct a higher level graph: JT

        # construct atom features of JT:
        # The modules are designed to deal with graphs with pad nodes, so a pad node should be added into the JT graph.
        batch_size, FP_size = activated_mol_FP1.size()
        pad_node_feature = t.zeros(batch_size, FP_size).to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
        JT_atom_features = t.stack([activated_mol_FP1, activated_mol_FP2, pad_node_feature], dim=1)
        # size: [batch_size, 3, FP_size]

        # JT construction complete.
        ###########################################################################################
        # branch JT:
        atom_FP_super = self.AtomEmbeddingJT(JT_atom_features,
                                             JT_bond_features,
                                             JT_atom_neighbor_list,
                                             JT_bond_neighbor_list)
        JT_FP, _ = self.MolEmbedding(atom_FP_super, JT_mask)

        entire_FP = t.cat([mol_FP1, mol_FP2, JT_FP, mol_FP_origin], dim=-1)
        # prediction = self.Classifier(entire_FP)
        return entire_FP

    def Input_cuda(self, Input):
        # to preprocess the input tensors
        [atom_features,
         bond_features,
         atom_neighbor_list_origin,
         bond_neighbor_list_origin,
         atom_mask_origin,
         atom_neighbor_list_changed,
         bond_neighbor_list_changed,
         frag_mask1,
         frag_mask2,
         bond_index,
         JT_bond_features,
         JT_atom_neighbor_list,
         JT_bond_neighbor_list,
         JT_mask] = Input

        # Tensor shapes are different in training and evaluation step
        # the shape of the input tensors is shown in featurizer.py
        if not self.training:
            atom_features = atom_features.squeeze(dim=0).to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            bond_features = bond_features.squeeze(dim=0).to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            atom_neighbor_list_origin = atom_neighbor_list_origin.squeeze(dim=0).to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            bond_neighbor_list_origin = bond_neighbor_list_origin.squeeze(dim=0).to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            atom_mask_origin = atom_mask_origin.squeeze(dim=0).to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))

            atom_neighbor_list_changed = atom_neighbor_list_changed.squeeze(dim=0).to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            bond_neighbor_list_changed = bond_neighbor_list_changed.squeeze(dim=0).to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            frag_mask1 = frag_mask1.squeeze(dim=0).to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            frag_mask2 = frag_mask2.squeeze(dim=0).to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            bond_index = bond_index.squeeze(dim=0).to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))

            JT_bond_features = JT_bond_features.squeeze(dim=0).to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            JT_atom_neighbor_list = JT_atom_neighbor_list.squeeze(dim=0).to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            JT_bond_neighbor_list = JT_bond_neighbor_list.squeeze(dim=0).to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            JT_mask = JT_mask.squeeze(dim=0).to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))

        else:
            atom_features = atom_features.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            bond_features = bond_features.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            atom_neighbor_list_origin = atom_neighbor_list_origin.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            bond_neighbor_list_origin = bond_neighbor_list_origin.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            atom_mask_origin = atom_mask_origin.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))

            atom_neighbor_list_changed = atom_neighbor_list_changed.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            bond_neighbor_list_changed = bond_neighbor_list_changed.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            frag_mask1 = frag_mask1.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            frag_mask2 = frag_mask2.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            bond_index = bond_index.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))

            JT_bond_features = JT_bond_features.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            JT_atom_neighbor_list = JT_atom_neighbor_list.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            JT_bond_neighbor_list = JT_bond_neighbor_list.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            JT_mask = JT_mask.to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))

        return [atom_features,
                bond_features,
                atom_neighbor_list_origin,
                bond_neighbor_list_origin,
                atom_mask_origin,
                atom_neighbor_list_changed,
                bond_neighbor_list_changed,
                frag_mask1,
                frag_mask2,
                bond_index,
                JT_bond_features,
                JT_atom_neighbor_list,
                JT_bond_neighbor_list,
                JT_mask]

class PyGFraGAT(nn.Module):
    def __init__(self, opt, FeatureExtractor = False):
        super(PyGFraGAT, self).__init__()
        self.BackboneATFP = PyGATFP(opt, FeatureExtractor = True)
        self.FeatureExtractor = FeatureExtractor
        self.Classifier = DNN(
            input_size=4 * opt.args['FPSize'],
            output_size=opt.args['OutputSize'],
            layer_sizes=opt.args['DNNLayers'],
            opt=opt
        )
        self.JTATFP = PyGATFP(opt, FeatureExtractor = True, JT=True)
        self.device = t.device(f"cuda:{opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')


    def forward(self, Input):
        if not self.training:
            # Transform the batch from mol-wise to graph-wise
            # each mol contains 1+singlebondnum*2 graphs for 12Layer
            batch_size = Input.y.size()[0]
            cumulated_singlebond_nums, cumulated_graph_nums = self.CalCumulatedGraphNumEval(Input)
            # for each molecule, if it have N singlebonds, it will generate N JT graphs, or 2*N frag graphs, with 1 original molecular graph
            # So, cumulated singlebond nums indicates the sum(Ni), responding to the number of graphs in Layer 3
            # and the cumulated graph nums indicates the sum(2*Ni + 1), responding to the number of grahs in Layer12

            Input.batch = self.EvalBatchTransform(Input, batch_size, cumulated_graph_nums)
            # Calculate Embeddings for 12Layer
            Embeddings_12 = self.BackboneATFP(Input)

            # Mask Embeddings if a molecule contains no singlebond
            Embeddings_12 = self.EvalMaskPadFrag(Embeddings_12, Input, batch_size, cumulated_graph_nums)

            # Build edge_index_JT
            edge_index_JT = self.EvalParseJTIndex(Input.JT__JT, batch_size, Input)

            # Build x_JT
            x_JT = self.EvalParsexJT(Embeddings_12, Input, batch_size)

            # Build mask_JT
            mask_JT = self.EvalBuildMaskJT(Input, batch_size)

            # build JT input
            input_JT = Data(x = x_JT, edge_index = edge_index_JT, edge_attr = Input.edge_attr_JT, batch=mask_JT)

            # cal JT embeddings
            Embeddings_3 = self.JTATFP(input_JT)

            # cal all graph num of JT
            graph_num = Input.singlebond_num.clone()
            graph_num[graph_num==0]=1
            all_JT_graphs_num = graph_num.sum()

            # eliminate the graph embedding of padnodes
            assert Embeddings_3.size()[0] >= all_JT_graphs_num
            Embeddings_3 = Embeddings_3[:all_JT_graphs_num]

            Embeddings_1 = []
            cumulated_graphs = 0
            for i in range(batch_size):
                cur_mol_Embeddings_1 = Embeddings_12[cumulated_graphs].unsqueeze(0)
                Embeddings_1.append(cur_mol_Embeddings_1)
                cumulated_graphs += (graph_num[i]*2+1)
            Embeddings_1 = t.cat(Embeddings_1,dim=0)

            cumulated_graphs = 0
            batch_sample_output = []
            for i in range(batch_size):
                cur_mol_sample_num = graph_num[i]
                cur_mol_1 = Embeddings_1[i].unsqueeze(0)
                cur_mol_1 = cur_mol_1.expand([cur_mol_sample_num,cur_mol_1.size()[-1]])

                cur_mol_2 = x_JT[cumulated_graphs*2:(cumulated_graphs*2+graph_num[i]*2),:].reshape(cur_mol_sample_num,-1)
                cur_mol_3 = Embeddings_3[cumulated_graphs:(cumulated_graphs+graph_num[i]),:]

                # print(f"cur_mol_1.size:{cur_mol_1.size()}")
                # print(f"cur_mol_2.size:{cur_mol_2.size()}")
                # print(f"cur_mol_3.size:{cur_mol_3.size()}")
                cur_mol_embedding = t.cat([cur_mol_1,cur_mol_2,cur_mol_3],dim=1)
                # print(f"cur_mol_embedding:{cur_mol_embedding[:,32:96]}")

                output = self.Classifier(cur_mol_embedding)
                output = output.mean(dim=0).unsqueeze(0)
                batch_sample_output.append(output)
                cumulated_graphs += graph_num[i]
                # print(f"output:{output}")
                # raise RuntimeError



                # cur_mol_1 = Embeddings_1[i]
                # cur_mol_embeddings = t.Tensor([]).to(self.device)
                # for j in range(graph_num[i]):
                #     cur_mol_cur_sample_1 = cur_mol_1.clone().unsqueeze(0)
                #     cur_mol_cur_sample_2 = x_JT[cumulated_graphs*2:(cumulated_graphs*2+2)].reshape(1,-1)
                #     cur_mol_cur_sample_3 = Embeddings_3[cumulated_graphs].unsqueeze(0)
                #     # print(f"embedding1:{cur_mol_cur_sample_1}")
                #     # print(f"embedding2:{cur_mol_cur_sample_2}")
                #     # print(f"embedding3:{cur_mol_cur_sample_3}")
                #     cur_mol_cur_sample_embedding = t.cat([cur_mol_cur_sample_1, cur_mol_cur_sample_2, cur_mol_cur_sample_3],dim=1)
                #     # cur_mol_cur_sample_embedding.unsqueeze_(0)
                #     # print(f"cur_mol_cur_sample_embedding.size():{cur_mol_cur_sample_embedding.size()}")
                #     cur_mol_embeddings = t.cat([cur_mol_embeddings, cur_mol_cur_sample_embedding], dim=0)
                #     cumulated_graphs += 1
                # print(f"cur_mol_embeddings.size():{cur_mol_embeddings.size()}")
                # print(f"cur_mol_embeddings:{cur_mol_embeddings[:,32:96]}")
                # output = self.Classifier(cur_mol_embeddings)
                # output = output.mean(dim=0).unsqueeze_(0)
                # print(output)
                # raise RuntimeError
                # batch_sample_output = t.cat([batch_sample_output, output],dim=0)
            batch_sample_output = t.cat(batch_sample_output, dim=0)

            return batch_sample_output



        # Transform the batch from mol-wise to graph-wise (each mol contains 3 graph for 12layer)
        batch_size = Input.y.size()[0]
        Input.batch = self.BatchTransform(Input, batch_size)

        # Calculate Embeddings for 12 Layers
        Embeddings_12 = self.BackboneATFP(Input)

        # Mask Embeddings if a molecule contains no singlebond
        Embeddings_12 = self.MaskPadFrag(Embeddings_12,Input, batch_size)

        # Build edge_index_JT
        edge_index_JT = self.ParseJTIndex(Input.JT__JT, batch_size)
        edge_index_JT = edge_index_JT.reshape(-1,2).T

        # Build x_JT
        x_JT = self.ParsexJT(Embeddings_12)

        # Build mask_JT
        mask_JT = self.BuildMaskJT(Input, batch_size)

        # build JT input
        input_JT = Data(x=x_JT, edge_index = edge_index_JT, edge_attr = Input.edge_attr_JT, batch = mask_JT)

        # cal JT embeddings
        Embeddings_3 = self.JTATFP(input_JT)
        assert Embeddings_3.size()[0] >= batch_size
        Embeddings_3 = Embeddings_3[:batch_size]


        Embeddings_12 = Embeddings_12.view(batch_size, -1)
        Embeddings_3 = Embeddings_3.view(batch_size, -1)
        Embeddings_all = t.cat([Embeddings_12, Embeddings_3], dim=1)


        output = self.Classifier(Embeddings_all)
        return output


    def CalCumulatedGraphNumEval(self,Input):
        singlebond_num = Input.singlebond_num
        graph_nums = singlebond_num.clone()
        graph_nums[graph_nums==0] = 1
        cumulated_singlebond_nums = graph_nums.cumsum(dim=0)
        graph_nums = graph_nums * 2 + 1
        cumulated_graph_nums = graph_nums.cumsum(dim=0)
        return cumulated_singlebond_nums, cumulated_graph_nums



    def EvalBuildMaskJT(self, Input, batch_size):
        # original mask JT without padnode eliminate
        graph_num = Input.singlebond_num.clone()
        graph_num[graph_num==0]=1
        # for i in range(batch_size):
        #     if Input.singlebond_num[i] == 0:
        #         graph_num[i] = 1
        all_JT_graphs_num = graph_num.sum()
        # print(f"all_JT_graphs_num:{all_JT_graphs_num}")
        mask_JT = t.arange(all_JT_graphs_num)
        mask_JT.unsqueeze_(1)
        mask_JT = mask_JT.expand(all_JT_graphs_num, 2)
        mask_JT = mask_JT.reshape(1,-1)
        mask_JT.squeeze_(0)
        # print(f"mask_JT:{mask_JT}")

        cumulated_nodes = 0
        for i in range(batch_size):
            if Input.singlebond_num[i] == 0:
                mask_JT[cumulated_nodes+1] = all_JT_graphs_num
            cumulated_nodes+=2*Input.singlebond_num[i]
        # print(f"mask_JT after eliminate:{mask_JT}")
        mask_JT = mask_JT.to(self.device)
        return mask_JT

    def BuildMaskJT(self, Input, batch_size):
        # original mask JT without padnode eliminate
        mask_JT = t.arange(batch_size)
        mask_JT.unsqueeze_(1)
        mask_JT = mask_JT.expand(batch_size, 2)
        mask_JT = mask_JT.reshape(1, -1)
        mask_JT.squeeze_(0)
        mask_JT = mask_JT.to(self.device)
        # mask_JT: [0,0,1,1,2,2,...,batch_size-1, batch_size-1]
        # each JT contains two node, so the original mask_JT is as above
        # print(f"mask_JT: {mask_JT}")

        # eliminate padnode from readout
        singlebond_num = Input.singlebond_num.unsqueeze(0)
        zeros = t.zeros([1,batch_size]).to(self.device)
        eliminate = t.cat([zeros, (singlebond_num==0)*1],dim=0)
        eliminate = eliminate.T
        eliminate = eliminate.reshape(-1)
        # eliminate: [ 0,0, 0,1, 0,0, 0,0,..., 0,0], if the second molecule in the batch is not breakable
        # i.e., the second node of JT of this molecule is a pad node and this node should not be readout

        reverse = -(mask_JT-(batch_size)).to(self.device)
        # reverse: [ batchsize, batchsize, batchsize-1, batchsize-1,...]
        # reverse indicates how many value should be added to the mask_JT element to make it to be (batch_size)

        bias = reverse * eliminate

        mask_JT = mask_JT + bias
        # print(f"mask_JT after eliminate the readout of padnode: {mask_JT}")
        return mask_JT.long()

    def EvalMaskPadFrag(self, Embeddings, Input, batch_size, cumulated_graph_nums):
        emb_size = Embeddings.size()[1]
        emb_nums = Embeddings.size()[0]
        mask = t.ones([emb_nums])
        graph_nums = Input.singlebond_num * 2 + 1
        graph_nums[Input.singlebond_num==0]=3

        bias = 0
        for i in range(batch_size):
            if Input.singlebond_num[i] == 0:
                mask[bias+2] = 0
            # print(f"bias:{bias}")
            bias += graph_nums[i]


        # print(f"mask:{mask}")
        mask.unsqueeze_(-1)
        mask = mask.expand([emb_nums, emb_size]).to(self.device)
        # print(f"mask.size():{mask.size()}")
        # print(f"Embeddings.size():{Embeddings.size()}")
        Embeddings = Embeddings * mask
        # print(f"Embeddings:{Embeddings}")
        return Embeddings

    def MaskPadFrag(self, Embeddings, Input, batch_size):
        emb_size = Embeddings.size()[1]
        index = t.arange(batch_size)
        index = index * 3
        index = index + 2
        # print(f"index: {index}")
        mask = t.ones([batch_size])
        mask[Input.singlebond_num==0] = 0
        mask.unsqueeze_(-1)
        mask = mask.expand([batch_size, emb_size]).to(self.device)
        Embeddings[index] = Embeddings[index] * mask
        # print(Embeddings)
        # print(Input.singlebond_num)
        # raise RuntimeError
        return Embeddings

    def EvalParseJTIndex(self, JT__JT, batch_size, Input):
        graph_num = Input.singlebond_num.clone()
        graph_num[graph_num==0]=1
        # for i in range(batch_size):
        #     if graph_num[i] == 0:
        #         graph_num[i] = 1
        edge_index_JT_before_pad = [JT__JT[(2*i):(2*i+2),0:(graph_num[i]*2)] for i in range(batch_size)]
        # print(edge_index_JT_before_pad)

        edge_index_all_JT = []
        cumulated_JT_node_num = 0
        for i in range(batch_size):
            cur_mol_edge_index_JT = edge_index_JT_before_pad[i]
            bias = t.arange(graph_num[i]) * 2
            bias.unsqueeze_(1)
            bias=bias.expand([graph_num[i],2])
            bias = bias.reshape(1,-1)
            bias = bias.expand([2,-1]).to(self.device)
            # print(f"cur_mol_edge_index_JT:{cur_mol_edge_index_JT}")
            # print(f"bias:{bias}")
            cur_mol_edge_index_JT = cur_mol_edge_index_JT + bias
            # print(f"cur_mol_edge_index_JT after bias:{cur_mol_edge_index_JT}")

            cur_mol_edge_index_JT = cur_mol_edge_index_JT + cumulated_JT_node_num
            edge_index_all_JT.append(cur_mol_edge_index_JT)
            # edge_index_all_JT = t.cat([edge_index_all_JT, cur_mol_edge_index_JT],dim=1)
            # print(f"edge_index_all_JT:{edge_index_all_JT}")
            cumulated_JT_node_num += graph_num[i] * 2
        edge_index_all_JT = t.cat(edge_index_all_JT, dim=1)
        return edge_index_all_JT.long()

    def ParseJTIndex(self, JT__JT,batch_size):
        JT__JT = JT__JT.view(batch_size, 2, -1)
        bias = t.arange(batch_size) * 2
        bias.unsqueeze_(-1).unsqueeze_(-1)
        bias = bias.expand(batch_size, 2, JT__JT.size()[-1]).to(self.device)
        # print(f"JT__JT:{JT__JT}")
        # print(f"bias:{bias}")
        edge_index_JT = JT__JT+bias
        # print(f"edge_index_JT:{edge_index_JT}")
        # raise RuntimeError
        return edge_index_JT.long()

    def EvalParsexJT(self, Embeddings, Input, batch_size):
        # print(f"Embeddings.size:{Embeddings.size()}")
        graph_num = Input.singlebond_num.clone()
        graph_num[graph_num==0] = 1
        cumulated_nodes = 0
        x_JT = []
        # x_JT = t.Tensor([]).to(self.device)
        for i in range(batch_size):
            # if graph_num[i] == 0:
            #     graph_num[i] = 1
            cur_mol_x_JT = Embeddings[(cumulated_nodes+1):(cumulated_nodes+1+graph_num[i]*2),:]
            x_JT.append(cur_mol_x_JT)
            # x_JT = t.cat([x_JT, cur_mol_x_JT],dim=0)
            cumulated_nodes += graph_num[i]*2 + 1
        x_JT = t.cat(x_JT,dim=0)
        # print(f"x_JT.size:{x_JT.size()}")
        # print(f"x_JT[-1]:{x_JT[-1]}")
        return x_JT




    def ParsexJT(self, Embeddings):
        index = t.arange(Embeddings.size()[0])
        index = index[index%3!=0]
        x_JT = Embeddings[index]
        # print(f"x_JT:{x_JT}")
        return x_JT

    def EvalBatchTransform(self, Input, batch_size, cumulated_graph_nums):
        # mask_12_before_pad = [Input.mask_12[i, 0:Input.mask_12_length[i]] for i in range(batch_size)]
        # mask_12_all = t.cat(mask_12_before_pad, dim=0)
        mask_12_all = Input.mask_12[Input.mask_12!=-1]
        # print(f"Input:{Input}")
        # print(f"mask_12:{Input.mask_12}")
        # if 55179 in Input.idx:
        #     print(f"mask_12:{Input.mask_12}")
        # print(f"Input.idx:{Input.idx}")
        # print(f"mask_12_all.size():{mask_12_all.size()}")

        # batch = Input.batch.clone()
        # graph_num = Input.singlebond_num*2 + 1
        # graph_num[Input.singlebond_num==0]=3
        #
        # bias = 0
        # # print(f"graph_num:{graph_num}")
        # for i in range(batch_size):
        #     batch[Input.batch==i] = bias
        #     # print(bias)
        #     bias += graph_num[i]
        # batch = mask_12_all + batch

        batch = Input.batch.clone()
        for i in range(1,batch_size):
            batch[Input.batch==i] = cumulated_graph_nums[i-1]

        # print(f"batch.size():{batch.size()}")
        # print(f"Input.x.size:{Input.x.size()}")
        batch = mask_12_all + batch

        return batch




    def BatchTransform(self, Input, batch_size):
        mask_12_before_pad = [Input.mask_12[i, 0:Input.mask_12_length[i]] for i in range(batch_size)]
        # print(f"mask_12_before_pad:{mask_12_before_pad}")
        mask_12_all = t.cat(mask_12_before_pad, dim = 0)
        # print(mask_12_all.size())
        # print(f"mask_12_all:{mask_12_all}")
        # print(f"Input.batch:{Input.batch}")

        # raise RuntimeError
        batch = Input.batch * 3
        batch = mask_12_all + batch
        # print(f"batch:{batch}")
        assert batch.max() + 1 == batch_size * 3

        return batch

    ##########

    def SplitFragEmbeddings(self, FragEmbeddings, Input_mol):
        cur_batch_size = Input_mol.batch.max() + 1
        single_bond_num = Input_mol.singlebond_num
        ptm = 0
        splitted_embeddings = []
        x_JT = []
        for i in range(cur_batch_size):
            if single_bond_num[i] != 0:
                cur_mol_frag_embeddings = FragEmbeddings[ptm:(ptm+2)]
                # print(f"cur_mol_frag_embeddings.size:{cur_mol_frag_embeddings.size()}")
                splitted_embeddings.append(cur_mol_frag_embeddings.unsqueeze(0))
                x_JT.append(cur_mol_frag_embeddings)
                ptm+=2
            else:
                cur_mol_frag_embeddings1 = FragEmbeddings[ptm:(ptm+1)]
                cur_mol_frag_embeddings2 = t.zeros([1,cur_mol_frag_embeddings1.size()[1]]).to(self.device)
                cur_mol_frag_embeddings = t.cat([cur_mol_frag_embeddings1,cur_mol_frag_embeddings2],dim=0)
                # print(f"cur_mol_frag_embeddings.size:{cur_mol_frag_embeddings.size()}")
                splitted_embeddings.append(cur_mol_frag_embeddings.unsqueeze(0))
                x_JT.append(cur_mol_frag_embeddings1)
                ptm+=1

        SplittedEmbeddings = t.cat(splitted_embeddings, dim=0)
        # print(f"x_JT: {x_JT}")
        # for item in x_JT:
        #     print(f"item.size: {item.size()}")
        x_JT = t.cat(x_JT, dim=0)
        # print(f"x_JT.size:{x_JT.size()}")

        return SplittedEmbeddings, x_JT

    def FragmentJTLayersInputForTraining(self, X_mol, edge_attr_mol, edge_index_mol, atom_nums, bond_nums, singlebond_nums, single_bond_list, cur_batch_size):
        # Build Input For Fragment Layer, for training phase.
        # 1) each mol graph is expanded to two subgraphs
        # 2)

        # X are not changed
        X_frags = X_mol.clone()

        # one edge will be popped for each breakable mol to generate a new edge_attr_frags
        edge_attr_frags = edge_attr_mol.clone()
        # one edge will be popped for each breakable mol to generate a new edge_index_frags
        edge_index_frags = edge_index_mol.clone()
        # a mask to distinguish different components of the molecules
        mask_frags = (t.ones([1, X_mol.size()[0]]) * -1).long()              # [1, all_atom_num]
        # print(f"mask_frags.size: {mask_frags.size()}")

        # print(f"singlebond_nums: {singlebond_nums}")
        # singlebond_tmp = singlebond_nums[1].clone()
        # singlebond_nums[1] = 0
        #
        cumulated_single_bonds = 0
        cumulated_atom_num = 0
        cumulated_bond_num = 0
        cumulated_component_num = 0



        X_JT = []
        edge_index_JT = []
        edge_attr_JT = []
        mask_JT = []
        cumulated_JT_num = 0
        cumulated_JT_node_num = 0

        for i in range(cur_batch_size):
            # atom_num, bond_num, singlebond_num of current molecule
            cur_mol_atom_num = atom_nums[i]
            cur_mol_bond_num = bond_nums[i]
            cur_mol_singlebond_num = singlebond_nums[i]

            if cur_mol_singlebond_num != 0:
                # Breakable

                # cut one bond
                # print(f"cumulated_single_bonds:{cumulated_single_bonds}")
                cur_mol_singlebond_list = single_bond_list[
                                          cumulated_single_bonds:(cumulated_single_bonds + cur_mol_singlebond_num)]
                edge_index_frags, edge_attr_frags, mask_frags, cumulated_component_num, removed_edge_attr = self.CutOneBond(
                    edge_index_frags,
                    edge_attr_frags,
                    cur_mol_singlebond_list,
                    cur_mol_singlebond_num,
                    cumulated_atom_num,
                    cumulated_bond_num,
                    mask_frags,
                    cumulated_component_num)

                # Build JT
                # To build JT graph, x, edge_attr, edge_index, batch_mask are required
                # x will be built after the FragEmbedding is calculated
                # other tensors will be built here
                # cumulated_JT_num: number of JT graphs have been calculated

                # A two node JT: edge_attr: [1, edge_size]
                #                edge_index: [[0,1],[1,0]]
                #                mask: [0,0]
                # Both need to be biased by cumulated_JT_num

                # edge_index, mask:
                cur_mol_edge_index_JT, cur_mol_mask_JT = self.CreateJunctionTree(cumulated_JT_node_num, cumulated_JT_num)
                edge_index_JT.append(cur_mol_edge_index_JT)
                edge_attr_JT.append(removed_edge_attr)
                edge_attr_JT.append(removed_edge_attr)
                mask_JT.append(cur_mol_mask_JT)
                #


                # cumulated nums increase
                cumulated_single_bonds += cur_mol_singlebond_num
                cumulated_atom_num += cur_mol_atom_num
                cumulated_bond_num += (cur_mol_bond_num - 2)        # since 2 bonds have been removed from edge_attr and edge_index,
                                                                    # the cumulated bond num should add cur_mol_bond_num - 2
                cumulated_JT_num += 1
                cumulated_JT_node_num += 2
            else:
                # no bond to cut

                # generate mask
                mask_tmp = t.ones([1, cur_mol_atom_num]) * cumulated_component_num
                mask_frags[0, cumulated_atom_num:(cumulated_atom_num + cur_mol_atom_num)] = mask_tmp
                cumulated_component_num += 1

                # Build JT
                # if breakable bond = 0
                # x: only add one graph embedding
                # edge_index: no edge
                # edge_attr: no edge attr
                # mask: one node
                # cumulated_JT_num:
                cur_mol_mask_JT = (t.ones([1, 1]) * cumulated_JT_num).long()
                mask_JT.append(cur_mol_mask_JT)

                cumulated_single_bonds += cur_mol_singlebond_num
                # cumulated_single_bonds += singlebond_tmp
                cumulated_atom_num += cur_mol_atom_num
                cumulated_bond_num += cur_mol_bond_num
                cumulated_JT_num += 1
                cumulated_JT_node_num += 1

        input_frags = Data(x = X_frags, edge_index = edge_index_frags, edge_attr = edge_attr_frags, batch = mask_frags.squeeze(0))

        edge_index_JT = t.cat(edge_index_JT,dim=1)
        edge_attr_JT = t.cat(edge_attr_JT, dim=0)
        mask_JT = t.cat(mask_JT, dim=1).squeeze(0)
        input_JT = Data(x = t.Tensor(X_JT), edge_index = edge_index_JT, edge_attr = edge_attr_JT,
                        batch = mask_JT)

        return input_frags, input_JT

    def FragmentJTLayersInputForEvaluation(self, X_mol, edge_attr_mol, edge_index_mol, atom_nums, bond_nums, singlebond_nums, single_bond_list, cur_batch_size):
        # For training, we use a 'remove' method, to remove edges from original batched graph
        # While for training, the batched graph will be greatly enlarged, so that we cannot use 'remove' method, but can only use a 'bulid' method
        # For each sample of each molecule, add the corresponding x, edge_attr, edge_index, mask to build overall batched tensors

        X_frags = t.Tensor([])
        edge_attr_frags = t.Tensor([])
        edge_index_frags = t.Tensor([])
        graph_mask_frags = t.Tensor([])
        mol_mask_frags = t.Tensor([])

        X_JT = t.Tensor([])
        edge_attr_JT = t.Tensor([])
        edge_index_JT = t.Tensor([])
        graph_mask_JT = t.Tensor([])
        mol_mask_JT = t.Tensor([])

        print(f"singlebond_nums: {singlebond_nums}")

        cumulated_atoms = 0
        cumulated_mol_atoms = 0
        cumulated_bonds = 0
        cumulated_mol_bonds = 0
        # cumulated_atoms/bonds: cumulated number of atoms/bonds in the generated batched tensor
        # cumulated_mol_atoms/bonds: cumulated number of atoms/bonds of the original batched tensor (X_mol, edge_attr_mol, edge_index_mol)

        cumulated_singlebonds = 0
        cumulated_frag_graphs = 0
        cumulated_JT_graphs = 0
        cumulated_JT_nodes = 0

        for i in range(cur_batch_size):
            # atom_num, bond_num, singlebond_num of current molecule
            cur_mol_atom_num = atom_nums[i]
            cur_mol_bond_num = bond_nums[i]
            cur_mol_singlebond_num = singlebond_nums[i]

            # extract tensors of current molecule from batched tensors
            cur_mol_x = X_mol[cumulated_mol_atoms:(cumulated_mol_atoms+cur_mol_atom_num),:]
            cur_mol_edge_attr = edge_attr_mol[cumulated_mol_bonds:(cumulated_mol_bonds+cur_mol_bond_num),:]
            cur_mol_edge_index = edge_index_mol[:,cumulated_mol_bonds:(cumulated_mol_bonds+cur_mol_bond_num)]
            cur_mol_edge_index = cur_mol_edge_index - cumulated_mol_atoms

            # print(f"cur_mol_edge_attr.size():{cur_mol_edge_attr.size()}")
            # print(f"cur_mol_edge_index:{cur_mol_edge_index}")
            # print(f"cur_mol_edge_index.size():{cur_mol_edge_index.size()}")
            # print(f"cur_mol_atom_num: {cur_mol_atom_num}")

            if cur_mol_singlebond_num != 0:
                # breakable
                # break each bond of the single bond list
                cur_mol_singlebonds = single_bond_list[cumulated_singlebonds:(cumulated_singlebonds + cur_mol_singlebond_num)]
                for singlebond in cur_mol_singlebonds:
                    # build Frags
                    # print(f"singlebond: {singlebond}")
                    [bond_idx, start_atom, end_atom] = singlebond
                    bond_idx = bond_idx * 2

                    cur_mol_cur_frag_x = cur_mol_x.clone()
                    cur_mol_cur_frag_edge_attr = cur_mol_edge_attr.clone()
                    cur_mol_cur_frag_edge_index = cur_mol_edge_index.clone()

                    cur_mol_cur_frag_mask = (t.ones([1,cur_mol_cur_frag_x.size()[0]]) * -1).long()

                    # cur_mol_cur_frag_xxx  are tensors for the sample with one
                    cur_mol_cur_frag_edge_index, cur_mol_cur_frag_edge_attr, cur_mol_cur_frag_removed_edge_attr = \
                        self.CutBond(cur_mol_cur_frag_edge_index, cur_mol_cur_frag_edge_attr, bond_idx, start_atom, end_atom)

                    # generate masks of current fragments
                    cur_mol_cur_frag_mask = self.ComponentSearch(cur_mol_cur_frag_edge_index, start_atom, cur_mol_cur_frag_mask, cumulated_frag_graphs)
                    cumulated_frag_graphs += 1
                    cur_mol_cur_frag_mask = self.ComponentSearch(cur_mol_cur_frag_edge_index, end_atom, cur_mol_cur_frag_mask, cumulated_frag_graphs)
                    cumulated_frag_graphs += 1

                    X_frags, edge_index_frags, edge_attr_frags, _ = self.CatGraphs(X_frags,
                                                                                   cur_mol_cur_frag_x,
                                                                                   edge_attr_frags,
                                                                                   cur_mol_cur_frag_edge_attr,
                                                                                   edge_index_frags,
                                                                                   cur_mol_cur_frag_edge_index)
                    # print(f"edge_index_frags:{edge_index_frags}")
                    graph_mask_frags = t.cat([graph_mask_frags, cur_mol_cur_frag_mask],dim=1)

                    cumulated_atoms += cur_mol_atom_num
                    cumulated_bonds += (cur_mol_bond_num - 2)


                    # Build JT   ？？？？？做到这里
                    cur_mol_edge_index_JT, cur_mol_mask_JT = self.CreateJunctionTree(cumulated_JT_nodes)
                    edge_index_JT.append(cur_mol_edge_index_JT)
                    edge_attr_JT.append(cur_mol_cur_frag_removed_edge_attr)
                    edge_attr_JT.append(cur_mol_cur_frag_removed_edge_attr)
                    mask_JT.append(cur_mol_mask_JT)



                assert X_frags.size()[0] == cumulated_atoms
                assert edge_index_frags.size()[1] == cumulated_bonds
                assert edge_attr_frags.size()[0] == cumulated_bonds
                assert graph_mask_frags.size()[1] == cumulated_atoms
                assert graph_mask_frags.max() == cumulated_frag_graphs - 1
                # print(f"X_frags:{X_frags}")
                # print(f"edge_index_frags:{edge_index_frags}")
                # print(f"edge_attr_frags:{edge_attr_frags}")
                # print(f"graph_mask_frags:{graph_mask_frags}")

                cur_mol_mask = (t.ones(1,cumulated_atoms) * i).long()
                mol_mask_frags = t.cat([mol_mask_frags, cur_mol_mask],dim=1)
                # print(f"mol_mask_frags:{mol_mask_frags}")
                raise RuntimeError







            # else:
                # no bond to break

    def CutOneBond(self, edge_index_origin, edge_attr_origin, singlebond_list, singlebond_num, atom_bias, bond_bias, mask_frags, component_bias):
        # edge_index_origin: batched edge_index
        # edge_attr_origin: batched edge_attr
        # singlebond_list: current mol singlebonds
        # singlebond_num: current mol singblebond number

        # randomly choose one bond to break
        chosen_idx = random.randint(0, singlebond_num-1)
        [bond_idx, start_atom, end_atom] = singlebond_list[chosen_idx]
        # the bond_idx and atom_idx are indices in original mol graph
        # They should be biased to find the location in batched tensors

        # remember that the bond_bias is directional, i.e. it corresponds to edge_attr.shape[0]
        # while the bond_idx is undirectional, i.e. the number of bond in the molecular graph calculated by rdkit
        # so, the bond_idx should be double and then added by bond_bias
        bond_idx_biased = bond_idx * 2 + bond_bias
        start_atom_biased = start_atom + atom_bias
        end_atom_biased = end_atom + atom_bias

        # remove one bond
        edge_index, edge_attr, removed_edge_attr = self.CutBond(edge_index_origin,
                                                                edge_attr_origin,
                                                                bond_idx_biased,
                                                                start_atom_biased,
                                                                end_atom_biased)

        # The edge_index and the edge_attr have been modified to remove the selected bond
        # And a mask to distinguish two components should be generated
        # Since the edge_index and edge_attr are batched, so the generated mask should not be 0-1, but should be biased by the number of components

        # generate mask
        mask_frags = self.ComponentSearch(edge_index, start_atom_biased, mask_frags, component_bias)
        component_bias += 1
        mask_frags = self.ComponentSearch(edge_index, end_atom_biased, mask_frags, component_bias)
        component_bias += 1

        return edge_index, edge_attr, mask_frags, component_bias, removed_edge_attr.unsqueeze(0)

    def CutAllBond(self, cur_mol_edge_index, cur_mol_edge_attr, cur_mol_singlebond_list, cur_mol_singlebond_num, atom_bias, bond_bias, mask_frags, component_bias):
        #
        AllFragEdgeIndex = []
        AllFragEdgeAttr = []
        # AllJT
        AllBondIdx = []


        for item in cur_mol_singlebond_list:
            edge_index_tmp = cur_mol_edge_index.clone()
            edge_attr_tmp = cur_mol_edge_attr.clone()
            # extract bond information
            [bond_idx, start_atom, end_atom] = item
            # bond_idx_biased = bond_idx + bond_bias
            # start_atom_biased = start_atom + atom_bias
            # end_atom_biasd = end_atom + atom_bias

            # remove bond
            edge_index_tmp, edge_attr_tmp, removed_edge_attr = self.CutBond(edge_index_tmp, edge_attr_tmp, bond_idx, start_atom, end_atom)
            AllFragEdgeAttr.append(edge_attr_tmp)
            AllFragEdgeIndex.append(edge_index_tmp)

    def CutBond(self, edge_index_ori, edge_attr_ori, bond_idx, start_atom, end_atom):
        # To remove a bond with [start_atom, end_atom] in edge_index, and pick up its corresponding edge_attr
        # the atom_idx and bond_idx have been biased, which directly point the corresponding bond and atoms (since the edge_index is biased)

        edge_index = edge_index_ori.t().tolist()
        edge_attr = edge_attr_ori.clone()
        assert [start_atom, end_atom] in edge_index
        assert [end_atom, start_atom] in edge_index

        bond_idx_directed = [bond_idx, bond_idx + 1]
        assert edge_index[bond_idx_directed[0]] == [start_atom, end_atom]
        assert edge_index[bond_idx_directed[1]] == [end_atom, start_atom]

        edge_index.remove([start_atom, end_atom])
        edge_index.remove([end_atom, start_atom])
        edge_index = t.Tensor(edge_index).long().t()

        # By removing one bond, the following part will move upward, so that both bond_idx_directed[0] are used
        edge_attr, removed_edge_attr_1 = self.DeleteTensorRow(edge_attr, bond_idx_directed[0])
        edge_attr, removed_edge_attr_2 = self.DeleteTensorRow(edge_attr, bond_idx_directed[0])
        assert edge_attr.size()[0] == edge_index.size()[1]
        # print(edge_attr[bond_idx])
        # print(edge_attr[bond_idx_directed[0]])
        # print(edge_attr_ori[bond_idx+2])
        # print(edge_attr[bond_idx_directed[0]] == edge_attr_ori[bond_idx + 2])
        assert False not in (edge_attr[bond_idx_directed[0]] == edge_attr_ori[bond_idx + 2])
        assert False not in (removed_edge_attr_1 == removed_edge_attr_2)

        # since the removed bonds are sharing the same attr, only one edge_attr will be returned.
        return edge_index, edge_attr, removed_edge_attr_1

    def DeleteTensorRow(self, tensor, index, dim=0):
        if dim==0:
            t1 = tensor[0:index, :]
            t2 = tensor[index+1:,:]
            return t.cat([t1,t2],dim=0), tensor[index]
        else:
            raise NotImplementedError

    def ComponentSearch(self, edge_index, root_node, mask, cur_component_index):
        # edge_index: a batched edge_index
        # root_node: the start node for searching, corresponding to the node_index in the edge_index
        # mask: a 'not-completed' mask of with the entire number of atoms in the batch
        # cur_component_index: the index of the current component that should be written into the mask.

        candidate_set = []
        candidate_set.append(root_node)

        mask[0,root_node] = cur_component_index

        while len(candidate_set) > 0:
            node = candidate_set[0]
            candidate_set.pop(0)

            for i in range(len(edge_index[0])):
                if edge_index[0][i] == node:
                    neighbor = edge_index[1][i]
                    if mask[0,neighbor] == -1:
                        candidate_set.append(neighbor)
                        mask[0,neighbor] = cur_component_index

        # print(f"mask after component search: {mask}")
        return mask

    # def CreateJunctionTree(self, JT_node_bias, JT_graph_bias):
    #     edge_index_JT = t.Tensor([[JT_node_bias, JT_node_bias+1],[JT_node_bias+1, JT_node_bias]]).long()
    #     mask = (t.ones([1,2]) * JT_graph_bias).long()
    #     return edge_index_JT, mask

    def CatGraphs(self, X1, X2, edge_attr1, edge_attr2, edge_index1, edge_index2):
        atom_num1 = X1.size()[0]
        atom_num2 = X2.size()[0]
        bond_num1 = edge_attr1.size()[0]
        bond_num2 = edge_attr2.size()[0]

        X = t.cat([X1, X2], dim=0)
        edge_attr = t.cat([edge_attr1, edge_attr2], dim=0)
        assert X.size()[0] == (atom_num1 + atom_num2)
        assert edge_attr.size()[0] == (bond_num1 + bond_num2)

        edge_index2 = edge_index2 + atom_num1

        edge_index = t.cat([edge_index1, edge_index2], dim=1).long()
        assert edge_index.size()[1] == (bond_num1 + bond_num2)

        mask1 = t.zeros([1,atom_num1])
        mask2 = t.zeros([1,atom_num2])
        mask = t.cat([mask1, mask2],dim=1).long()

        return X, edge_index, edge_attr, mask




#########################################################################
# Test Codes
#########################################################################