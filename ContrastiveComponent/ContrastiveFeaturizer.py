import torch as t
import numpy as np
from TrainingFramework.ChemUtils import *
from TrainingFramework.Featurizer import BasicFeaturizer
from FragComponents.Utils import *
from torch_geometric.data import Data


class PyGFraGATContraFeaturizer(BasicFeaturizer):
    def __init__(self, opt):
        super(PyGFraGATContraFeaturizer, self).__init__()
        self.opt = opt
        self.stored_dataset = {}

    def featurize(self, item):
        SMILES = item
        mol = GetMol(SMILES)

        # Calculate atom/bond features (x, edge_attr)
        if self.opt.args['FeatureCategory'] == 'BaseOH':
            x, edge_attr = GetBaseFeatureOH(mol)
        elif self.opt.args['FeatureCategory'] == 'BaseED':
            x, edge_attr = GetBaseFeatureED(mol)
        elif self.opt.args['FeatureCategory'] == 'OGB':
            x, edge_attr = GetOGBFeature(mol)
        elif self.opt.args['FeatureCategory'] == 'RichOH':
            x, edge_attr = GetRichFeatureOH(mol)
        elif self.opt.args['FeatureCategory'] == 'RichED':
            x, edge_attr = GetRichFeatureED(mol)
        else:
            raise NotImplementedError('No feature category.')

        x_orimol = t.Tensor(x)
        edge_attr_orimol = t.Tensor(edge_attr)
        # Calculate topology
        edge_index_orimol = t.Tensor(GetEdgeList(mol, bidirection = True)).long().t()
        data_orimol = Data(x = x_orimol,
                           edge_attr = edge_attr_orimol,
                           edge_index = edge_index_orimol)

        # x: [atom_num, atom_feature_size]
        # edge_attr: [bond_num, bond_feature_size]
        # edge_index: [2, bond_num]

        atom_num = x_orimol.size()[0]
        bond_num = edge_attr_orimol.size()[0]

        # Calculate Singlebonds
        singlebond_list = GetSingleBonds(mol)
        singlebond_num = len(singlebond_list)

        if singlebond_num == 0:
            return None

        else:
        # Generate Graph Tensors for Frags
            x_all_frags, edge_index_all_frags, edge_attr_all_frags, mask_all_frags = \
                 self.GenerateFrags(x_orimol, edge_attr_orimol, edge_index_orimol, singlebond_list,
                                             singlebond_num)
            mask_frag_length = mask_all_frags.size()[1]
            mask_all_frags = self.PadMask(mask_all_frags)

            data_frag = Data(SMILES = SMILES,
                    x = x_all_frags,
                    edge_index = edge_index_all_frags,
                    edge_attr = edge_attr_all_frags,
                    mask_frags = mask_all_frags,
                    atom_num = atom_num,
                    bond_num = bond_num,
                    singlebond_num = singlebond_num,
                    mask_frag_length = mask_frag_length)

            return data_frag, data_orimol, SMILES


    def GenerateFrags(self, x_orimol, edge_attr_orimol, edge_index_orimol, singlebond_list, singlebond_num):
        # singlebond_num != 0, break the molecule into different pair of fragments
        # generate views
        x_all_frags = t.Tensor([])
        edge_attr_all_frags = t.Tensor([])
        edge_index_all_frags = t.Tensor([])
        mask_all_frags = t.Tensor([])

        for i in range(singlebond_num):
            # copy a group of tensors
            x_frags = x_orimol.clone()
            edge_attr_frags = edge_attr_orimol.clone()
            edge_index_frags = edge_index_orimol.clone()

            # Get singlebond
            singlebond = singlebond_list[i]
            [bond_idx, start_atom, end_atom] = singlebond

            # remove the chosen bond from edge_attr and edge_index:
            edge_index_frags, edge_attr_frags, removed_edge_attr = self.CutOneBond(edge_index_frags, edge_attr_frags,
                                                                                   bond_idx, start_atom, end_atom)
            if removed_edge_attr.size()[0] != 1:
                removed_edge_attr.unsqueeze_(0)

            # generate mask to identify nodes belong to different fragments
            mask_frags = (t.ones([1, x_frags.size()[0]]) * -2).long()  # orimol_atom_num
            mask_frags = self.ComponentSearch(edge_index_frags, start_atom, mask_frags, 0)
            mask_frags = self.ComponentSearch(edge_index_frags, end_atom, mask_frags, 1)

            if -2 in mask_frags:
                mask_frags[mask_frags == -2] = 0

            # Till here, tensors for FragLayer have been done
            # x_frags: [atom_num, atom_feature_size]
            # edge_index_frags: [2, edge_num - 2]
            # edge_attr_frags: [edge_num - 2, bond_feature_size]
            # mask_frags: [1, atom_num]
            # edge_num = bond_num * 2
            # mask_frags are 0/1


            # Till here, Tensors for Frags and JT of the current sample have been done
            # Need to be added to the entire tensors of FragLayer and JT Layer
            x_all_frags, edge_index_all_frags, edge_attr_all_frags, mask_all_frags = \
                self.CatGraph(x_all_frags, x_frags, edge_attr_all_frags, edge_attr_frags, edge_index_all_frags,
                              edge_index_frags, mask1 = mask_all_frags, mask2 = mask_frags)

        return x_all_frags, edge_index_all_frags, edge_attr_all_frags, mask_all_frags

    def GetLabelFromValues(self, values):
        label = []
        if self.opt.args['ClassNum'] == 1:
            if self.opt.args['TaskNum'] == 1:
                label.append(float(values))
            else:
                for v in values:
                    label.append(float(v))
            label = t.Tensor(label)
        else:
            for v in values:
                label.append(int(v))
            label = t.Tensor(label).long()
        label.unsqueeze_(-1)

        return label

    def ComponentSearch(self, edge_index, root_node, mask, component_bias):
        # given a root node, find the connected part by edge_index
        # mask is a pre-constructed tensor with size [1, atom_num]
        # initially, mask are all -1
        # by two-times component search, mask will be 0 or 1

        candidate_set = []
        candidate_set.append(root_node)

        mask[0, root_node] = component_bias

        while len(candidate_set) > 0:
            node = candidate_set[0]
            candidate_set.pop(0)

            for i in range(len(edge_index[0])):
                if edge_index[0,i] == node:
                    neighbor = edge_index[1,i]
                    if mask[0, neighbor] == -2:
                        candidate_set.append(neighbor)
                        mask[0, neighbor] = component_bias

        return mask

    def CutOneBond(self, edge_index_origin, edge_attr_origin, bond_idx, start_atom, end_atom):
        # To remove a bond with [start_atom, end_atom] in edge_index, and pickup its corresponding edge_attr
        edge_index = edge_index_origin.t().tolist()
        edge_attr = edge_attr_origin.clone()
        assert [start_atom, end_atom] in edge_index
        assert [end_atom, start_atom] in edge_index

        # each bond_idx correspond to two edges
        bond_idx_directed = [bond_idx * 2, bond_idx * 2 + 1]
        assert edge_index[bond_idx_directed[0]] == [start_atom, end_atom]
        assert edge_index[bond_idx_directed[1]] == [end_atom, start_atom]

        # remove bond from edge_index
        edge_index.remove([start_atom, end_atom])
        edge_index.remove([end_atom, start_atom])
        edge_index = t.Tensor(edge_index).long().t()
        if edge_index.size()[0] == 0:
            edge_index = t.empty([2,0])

        # By removing one bond, the following part will move upward, so that both bond_idx_directed[0] are used
        edge_attr, removed_edge_attr1 = self.DeleteTensorRow(edge_attr, bond_idx_directed[0])
        edge_attr, removed_edge_attr2 = self.DeleteTensorRow(edge_attr, bond_idx_directed[0])
        # print(f"edge_attr.size:{edge_attr.size()}")
        # print(f"edge_index.size:{edge_index.size()}")

        assert edge_attr.size()[0] == edge_index.size()[1]
        # assert False not in (edge_attr[bond_idx_directed[0]] == edge_attr_origin[bond_idx_directed[0]+2])
        assert False not in (removed_edge_attr1 == removed_edge_attr2)
        return edge_index, edge_attr, removed_edge_attr1

    def DeleteTensorRow(self, tensor, index, dim=0):
        if dim==0:
            t1 = tensor[0:index,:]
            t2 = tensor[(index+1):,:]
            return t.cat([t1,t2],dim=0), tensor[index]
        else:
            return NotImplementedError

    def CatGraph(self, x1, x2, edge_attr1, edge_attr2, edge_index1, edge_index2, mask2, mask1=None):
        atom_num1 = x1.size()[0]
        atom_num2 = x2.size()[0]
        bond_num1 = edge_attr1.size()[0]
        bond_num2 = edge_attr2.size()[0]

        x = t.cat([x1,x2],dim=0)
        edge_attr = t.cat([edge_attr1, edge_attr2],dim=0)

        edge_index2 = edge_index2 + atom_num1

        edge_index = t.cat([edge_index1, edge_index2],dim=1).long()
        assert edge_index.size()[1] == edge_attr.size()[0]

        if mask1 == None:
            mask1 = t.zeros([1,atom_num1])
            mask2 = mask2+1
        else:
            if atom_num1 != 0:
                mask2 = mask2 + (mask1[0].max() + 1)

        mask = t.cat([mask1, mask2],dim=1).long()
        if atom_num2 != 0:
            assert mask.size()[1] == (atom_num1+atom_num2)

        return x, edge_index, edge_attr, mask

    def prefeaturize(self, dataset):
        # screened_dataset = []
        # discarded_dataset = []
        # self.max_atom_num = 0
        # self.max_singlebond_num = 0
        self.max_mask_num = 0
        for item in dataset:
            SMILES = item
            mol = GetMol(SMILES)
            atom_num = GetAtomNum(mol)
            singlebond_num = len(GetSingleBonds(mol))
            mask_num = atom_num * singlebond_num
            self.max_mask_num = max(mask_num, self.max_mask_num)

        self.max_mask_num += 1
        print(f"Max mask length:{self.max_mask_num}")
        # return screened_dataset

    def load_information(self, fp):
        import json
        self.max_mask_num = json.load(fp)

    def PadMask(self, mask_frag):
        mask_len = mask_frag.size()[1]
        pad_num = self.max_mask_num - mask_len
        pad = t.ones([1,pad_num])*-1
        pad_mask = t.cat([mask_frag, pad],dim=1).long()
        return pad_mask

class PyGFraGATContraFinetuneFeaturizer(BasicFeaturizer):
    def __init__(self, opt):
        super(PyGFraGATContraFinetuneFeaturizer, self).__init__()
        self.opt = opt

    def featurize(self, item):
        SMILES = item['SMILES']
        Value = item['Value']
        # idx = item['idx']
        idx = t.Tensor([item['idx']]).long()

        Label = self.GetLabelFromValues(Value)
        mol = GetMol(SMILES)

        # Calculate atom/bond features (x, edge_attr)
        if self.opt.args['FeatureCategory'] == 'BaseOH':
            x, edge_attr = GetBaseFeatureOH(mol)
        elif self.opt.args['FeatureCategory'] == 'BaseED':
            x, edge_attr = GetBaseFeatureED(mol)
        elif self.opt.args['FeatureCategory'] == 'OGB':
            x, edge_attr = GetOGBFeature(mol)
        elif self.opt.args['FeatureCategory'] == 'RichOH':
            x, edge_attr = GetRichFeatureOH(mol)
        elif self.opt.args['FeatureCategory'] == 'RichED':
            x, edge_attr = GetRichFeatureED(mol)
        else:
            raise NotImplementedError('No feature category.')

        x_orimol = t.Tensor(x)
        edge_attr_orimol = t.Tensor(edge_attr)
        # Calculate topology
        edge_index_orimol = t.Tensor(GetEdgeList(mol, bidirection = True)).long().t()
        data_orimol = Data(x = x_orimol,
                           edge_attr = edge_attr_orimol,
                           edge_index = edge_index_orimol,
                           y = Label.t(),
                           idx = idx)

        atom_num = x_orimol.size()[0]
        bond_num = edge_attr_orimol.size()[0]

        # Calculate Singlebonds
        singlebond_list = GetSingleBonds(mol)
        singlebond_num = len(singlebond_list)

        if singlebond_num == 0:
            x_frags = data_orimol.x.clone()
            edge_attr_frags = data_orimol.edge_attr.clone()
            edge_index_frags = data_orimol.edge_index.clone()

            # no bond to cut, the fragview is just one part, the entire molecule
            # As the Frags are packaged to a large graph and it is no need to follow the frag_layer structure
            # In FraSICL, we will not pad the other empty graph

            # pad_node_feature = t.zeros([1, x_frags.size()[1]])
            # x_frags = t.cat([x_frags, pad_node_feature], dim = 0)  # [orimol_atom_num + 1

            # x_frags, edge_index, edge_attr not need to change
            # build mask_frags
            mask_frags = t.zeros([1, x_frags.size()[0]]).long()
            # mask_frags[0, -1] = 1  # [0 * orimol_atom_num, 1]
            mask_frags_length = mask_frags.size()[1]
            mask_frags = self.PadMask(mask_frags)

            # Till here, tensors for FragLayer have been done
            # x_frags: [atom_num, atom_feature_size]
            # edge_index_frags: [2, edge_num]
            # edge_attr_frags: [edge_num, bond_feature_size]
            # mask_frags: [1, atom_num]
            # edge_num = bond_num * 2

            data_frag = Data(SMILES = SMILES,
                             x = x_frags,
                             edge_index = edge_index_frags,
                             edge_attr = edge_attr_frags,
                             y = Label.t(),
                             mask_frags = mask_frags,
                             atom_num = atom_num,
                             bond_num = bond_num,
                             singlebond_num = singlebond_num,
                             mask_frag_length = mask_frags_length,
                             idx = idx)

            return data_frag, data_orimol, SMILES

        else:
            # Generate Graph Tensors for Frags
            x_all_frags, edge_index_all_frags, edge_attr_all_frags, mask_all_frags = \
                self.GenerateFrags(x_orimol, edge_attr_orimol, edge_index_orimol, singlebond_list,
                                   singlebond_num)
            mask_frag_length = mask_all_frags.size()[1]
            mask_all_frags = self.PadMask(mask_all_frags)

            data_frag = Data(SMILES = SMILES,
                             x = x_all_frags,
                             edge_index = edge_index_all_frags,
                             edge_attr = edge_attr_all_frags,
                             y = Label.t(),
                             mask_frags = mask_all_frags,
                             atom_num = atom_num,
                             bond_num = bond_num,
                             singlebond_num = singlebond_num,
                             mask_frag_length = mask_frag_length,
                             idx = idx)

            return data_frag, data_orimol, SMILES

    def GenerateFrags(self, x_orimol, edge_attr_orimol, edge_index_orimol, singlebond_list, singlebond_num):
        # singlebond_num != 0, break the molecule into different pair of fragments
        # generate views
        x_all_frags = t.Tensor([])
        edge_attr_all_frags = t.Tensor([])
        edge_index_all_frags = t.Tensor([])
        mask_all_frags = t.Tensor([])

        for i in range(singlebond_num):
            # copy a group of tensors
            x_frags = x_orimol.clone()
            edge_attr_frags = edge_attr_orimol.clone()
            edge_index_frags = edge_index_orimol.clone()

            # Get singlebond
            singlebond = singlebond_list[i]
            [bond_idx, start_atom, end_atom] = singlebond

            # remove the chosen bond from edge_attr and edge_index:
            edge_index_frags, edge_attr_frags, removed_edge_attr = self.CutOneBond(edge_index_frags, edge_attr_frags,
                                                                                   bond_idx, start_atom, end_atom)
            if removed_edge_attr.size()[0] != 1:
                removed_edge_attr.unsqueeze_(0)

            # generate mask to identify nodes belong to different fragments
            mask_frags = (t.ones([1, x_frags.size()[0]]) * -2).long()  # orimol_atom_num
            mask_frags = self.ComponentSearch(edge_index_frags, start_atom, mask_frags, 0)
            mask_frags = self.ComponentSearch(edge_index_frags, end_atom, mask_frags, 1)

            if -2 in mask_frags:
                mask_frags[mask_frags == -2] = 0

            # Till here, tensors for FragLayer have been done
            # x_frags: [atom_num, atom_feature_size]
            # edge_index_frags: [2, edge_num - 2]
            # edge_attr_frags: [edge_num - 2, bond_feature_size]
            # mask_frags: [1, atom_num]
            # edge_num = bond_num * 2
            # mask_frags are 0/1

            # Till here, Tensors for Frags and JT of the current sample have been done
            # Need to be added to the entire tensors of FragLayer and JT Layer
            x_all_frags, edge_index_all_frags, edge_attr_all_frags, mask_all_frags = \
                self.CatGraph(x_all_frags, x_frags, edge_attr_all_frags, edge_attr_frags, edge_index_all_frags,
                              edge_index_frags, mask1 = mask_all_frags, mask2 = mask_frags)

        return x_all_frags, edge_index_all_frags, edge_attr_all_frags, mask_all_frags

    def GetLabelFromValues(self, values):
        label = []
        if self.opt.args['ClassNum'] == 1:
            if self.opt.args['TaskNum'] == 1:
                label.append(float(values))
            else:
                for v in values:
                    label.append(float(v))
            label = t.Tensor(label)
        else:
            for v in values:
                label.append(int(v))
            label = t.Tensor(label).long()
        label.unsqueeze_(-1)

        return label

    def ComponentSearch(self, edge_index, root_node, mask, component_bias):
        # given a root node, find the connected part by edge_index
        # mask is a pre-constructed tensor with size [1, atom_num]
        # initially, mask are all -1
        # by two-times component search, mask will be 0 or 1

        candidate_set = []
        candidate_set.append(root_node)

        mask[0, root_node] = component_bias

        while len(candidate_set) > 0:
            node = candidate_set[0]
            candidate_set.pop(0)

            for i in range(len(edge_index[0])):
                if edge_index[0, i] == node:
                    neighbor = edge_index[1, i]
                    if mask[0, neighbor] == -2:
                        candidate_set.append(neighbor)
                        mask[0, neighbor] = component_bias

        return mask

    def CutOneBond(self, edge_index_origin, edge_attr_origin, bond_idx, start_atom, end_atom):
        # To remove a bond with [start_atom, end_atom] in edge_index, and pickup its corresponding edge_attr
        edge_index = edge_index_origin.t().tolist()
        edge_attr = edge_attr_origin.clone()
        assert [start_atom, end_atom] in edge_index
        assert [end_atom, start_atom] in edge_index

        # each bond_idx correspond to two edges
        bond_idx_directed = [bond_idx * 2, bond_idx * 2 + 1]
        assert edge_index[bond_idx_directed[0]] == [start_atom, end_atom]
        assert edge_index[bond_idx_directed[1]] == [end_atom, start_atom]

        # remove bond from edge_index
        edge_index.remove([start_atom, end_atom])
        edge_index.remove([end_atom, start_atom])
        edge_index = t.Tensor(edge_index).long().t()
        if edge_index.size()[0] == 0:
            edge_index = t.empty([2, 0])

        # By removing one bond, the following part will move upward, so that both bond_idx_directed[0] are used
        edge_attr, removed_edge_attr1 = self.DeleteTensorRow(edge_attr, bond_idx_directed[0])
        edge_attr, removed_edge_attr2 = self.DeleteTensorRow(edge_attr, bond_idx_directed[0])
        # print(f"edge_attr.size:{edge_attr.size()}")
        # print(f"edge_index.size:{edge_index.size()}")

        assert edge_attr.size()[0] == edge_index.size()[1]
        # assert False not in (edge_attr[bond_idx_directed[0]] == edge_attr_origin[bond_idx_directed[0]+2])
        assert False not in (removed_edge_attr1 == removed_edge_attr2)
        return edge_index, edge_attr, removed_edge_attr1

    def DeleteTensorRow(self, tensor, index, dim = 0):
        if dim == 0:
            t1 = tensor[0:index, :]
            t2 = tensor[(index + 1):, :]
            return t.cat([t1, t2], dim = 0), tensor[index]
        else:
            return NotImplementedError

    def CatGraph(self, x1, x2, edge_attr1, edge_attr2, edge_index1, edge_index2, mask2, mask1 = None):
        atom_num1 = x1.size()[0]
        atom_num2 = x2.size()[0]
        bond_num1 = edge_attr1.size()[0]
        bond_num2 = edge_attr2.size()[0]

        x = t.cat([x1, x2], dim = 0)
        edge_attr = t.cat([edge_attr1, edge_attr2], dim = 0)

        edge_index2 = edge_index2 + atom_num1

        edge_index = t.cat([edge_index1, edge_index2], dim = 1).long()
        assert edge_index.size()[1] == edge_attr.size()[0]

        if mask1 == None:
            mask1 = t.zeros([1, atom_num1])
            mask2 = mask2 + 1
        else:
            if atom_num1 != 0:
                mask2 = mask2 + (mask1[0].max() + 1)

        mask = t.cat([mask1, mask2], dim = 1).long()
        if atom_num2 != 0:
            assert mask.size()[1] == (atom_num1 + atom_num2)

        return x, edge_index, edge_attr, mask

    def prefeaturize(self, dataset):
        # screened_dataset = []
        # discarded_dataset = []
        # self.max_atom_num = 0
        # self.max_singlebond_num = 0
        self.max_mask_num = 0
        for item in dataset:
            SMILES = item['SMILES']
            mol = GetMol(SMILES)
            atom_num = GetAtomNum(mol)
            singlebond_num = len(GetSingleBonds(mol))
            mask_num = atom_num * singlebond_num
            self.max_mask_num = max(mask_num, self.max_mask_num)

        self.max_mask_num += 1
        print(f"Max mask length:{self.max_mask_num}")
        # return screened_dataset

    def load_information(self, fp):
        import json
        self.max_mask_num = json.load(fp)

    def PadMask(self, mask_frag):
        mask_len = mask_frag.size()[1]
        pad_num = self.max_mask_num - mask_len
        pad = t.ones([1, pad_num]) * -1
        pad_mask = t.cat([mask_frag, pad], dim = 1).long()
        return pad_mask
################################################################################################################
# Test Codes
if __name__ == '__main__':
    SMILES1 = 'Cc1cc(C)c(C)cc1C'
    SMILES2 = 'c1ccccc1'
    SMILES3 = 'Cc1cc(C)c(C)cc1C'

    Value = '1'
    dataset = [{'SMILES': SMILES1, 'Value': Value}, {'SMILES': SMILES2, 'Value': Value},
               {'SMILES': SMILES3, 'Value': Value}]
    item1 = dataset[0]
    item2 = dataset[1]
    item3 = dataset[2]
    # mol1 = GetMol(SMILES1)
    # DrawMolGraph(mol1, './', '1')
    # mol2 = GetMol(SMILES2)
    # DrawMolGraph(mol2, './', '2')
    # mol3 = GetMol(SMILES3)
    # DrawMolGraph(mol3, './', '3')


    class OPT(object):
        def __init__(self):
            super(OPT, self).__init__()
            self.args = {'ClassNum': 1,
                         'TaskNum': 1,
                         'FeatureCategory': 'BaseOH'}


    opt = OPT()

    featurizer = PyGFraGATContraFeaturizer(opt)
    featurizer.prefeaturize(dataset)
    print(f"featurizing data1")
    data1 = featurizer.featurize(item1)
    print(f"featurizing data2")
    data2 = featurizer.featurize(item2)
    print(f"featurizing data3")
    data3 = featurizer.featurize(item3)
    # data1.__setattr__('idx',1)
    # data2.__setattr__('idx',2)
    # data3.__setattr__('idx',3)

    print(f"data1:{data1}")
    print(f"data2:{data2}")
    print(f"data3:{data3}")

    dataset = [data1, data2, data3]
    from torch_geometric.loader import DataLoader

    loader = DataLoader(dataset, batch_size = 3, shuffle = True)

    for batch in loader:
        print(batch)


