from TrainingFramework.FileUtils import *
from TrainingFramework.Splitter import *
from TrainingFramework.Featurizer import *
# todo(zqzhang): updated in FraContra
from ContrastiveComponent.ContrastiveFeaturizer import PyGFraGATContraFeaturizer,PyGFraGATContraFinetuneFeaturizer
from torch.utils import data
from torch_geometric.data import InMemoryDataset
import os
from tqdm import tqdm
########################################################################################
class MolDataset(data.Dataset):
    # Mol Dataset Module.
    def __init__(self, dataset, opt, mode, trainset_part = None):
        super(MolDataset, self).__init__()
        self.dataset = dataset
        self.opt = opt
        self.mode = mode

        self.BuildFeaturizer()

        # if use methods in Attentive FP or FraGAT to construct dataset, some more works should be done here.
        if (opt.args['Feature'] == 'AttentiveFP') or (opt.args['Feature'] == 'FraGAT'):
            print('Prefeaturizing molecules...')
            self.featurizer.GetPad(self.dataset)
            self.prefeaturized_dataset = self.featurizer.prefeaturize(self.dataset)
            # Both ATFP and FraGAT featurizers have function GetPad and prefeaturize.
            print("Prefeaturization complete.")

        if (opt.args['Feature'] == 'PyGFraGAT'):
            print('Prefeaturizing...')
            self.featurizer.prefeaturize(self.dataset)
            # for evaluation of FraGAT, calculating is time consuming
            # store the calculated data for the next time calling
            if self.mode == 'EVAL':
                self.stored_dataset = {}
                self.PreCalculating()

        # todo(zqzhang): updated in FraContra
        if opt.args['Feature'] == 'FraContra':
            # Feature == FraContra, indicates this script is for Large Dataset Pretraining
            # Prefeaturizing to compute the max_mask_num (for frag views, max_mask_num is required)
            # If no file exists, computing and save the value; otherwise, load the value from file.
            # As the splitting uses random seed, different seed lead to different Train/Valid set.
            # So, the pre-computed values and Tensors are seed-related. Must be distinguished by seeds.
            if not os.path.exists(opt.args['PretrainDataPath']+f'_PrefeaturizeInformation_{mode}_seed{opt.args["PretrainDatasetValidSeed"]}_{opt.args["PretrainValidSplitRate"]}.json'):
                print(f"Prefeaturizing for pretrain dataset...")
                self.featurizer.prefeaturize(self.dataset)
                print(f"saving prefeaturize information...")
                with open(opt.args['PretrainDataPath']+f'_PrefeaturizeInformation_{mode}_seed{opt.args["PretrainDatasetValidSeed"]}_{opt.args["PretrainValidSplitRate"]}.json','w') as f:
                    json.dump(self.featurizer.max_mask_num, f)
            else:
                print(f"Loading prefeaturize information...")
                with open(opt.args['PretrainDataPath']+f'_PrefeaturizeInformation_{mode}_seed{opt.args["PretrainDatasetValidSeed"]}_{opt.args["PretrainValidSplitRate"]}.json','r') as f:
                    self.featurizer.load_information(f)

            # For the Valid set for FraContra, store its tensors for speed up the training.
            if self.mode == 'EVAL':
                # if no file exists, compute the tensors, store it, and save the Tensors in .pt file
                # otherwise, load the Tensors from .pt file and
                if os.path.exists(opt.args['PretrainDataPath']+f"_PreCalculatedTensors_seed{opt.args['PretrainDatasetValidSeed']}_{opt.args['PretrainValidSplitRate']}.pt"):
                    print(f"Loading stored dataset...")
                    self.stored_dataset = t.load(opt.args['PretrainDataPath']+f"_PreCalculatedTensors_seed{opt.args['PretrainDatasetValidSeed']}_{opt.args['PretrainValidSplitRate']}.pt")
                else:
                    print(f"Precalculating features of items for eval set...")
                    self.stored_dataset = {}
                    self.PreCalculating()
                    t.save(self.stored_dataset, opt.args['PretrainDataPath']+f"_PreCalculatedTensors_seed{opt.args['PretrainDatasetValidSeed']}_{opt.args['PretrainValidSplitRate']}.pt")

            if self.mode == 'TRAIN':
                if self.opt.args['SplittingTrainset']:
                    self.TrainsetPart = trainset_part
                    if os.path.exists(opt.args['PretrainDataPath']+f"_PreCalculatedTensors_seed{opt.args['PretrainDatasetValidSeed']}_{opt.args['PretrainValidSplitRate']}_Trainset_Part_{trainset_part}.pt"):
                        # print(f"Loading stored datset...")
                        # self.stored_dataset = t.load(opt.args['PretrainDataPath']+f"_PreCalculatedTensors_seed{opt.args['PretrainDatasetValidSeed']}_{opt.args['PretrainValidSplitRate']}_Trainset_Part_{trainset_part}.pt")
                        print(f"Precalculated dataset exists. Will be loaded when calling...")
                        self.stored_dataset = {}
                    else:
                        print(f"Precalculating tensors of items for training set part {trainset_part}...")
                        self.stored_dataset = {}
                        self.PreCalculating()
                        t.save(self.stored_dataset, opt.args['PretrainDataPath']+f"_PreCalculatedTensors_seed{opt.args['PretrainDatasetValidSeed']}_{opt.args['PretrainValidSplitRate']}_Trainset_Part_{trainset_part}.pt")
                        self.stored_dataset = {}

        if (opt.args['Feature'] == 'FraContra-finetune'):
            print('Prefeaturizing...')
            self.featurizer.prefeaturize(self.dataset)
            # for evaluation of FraGAT, calculating is time consuming
            # store the calculated data for the next time calling
            # if self.mode == 'EVAL':
            #     self.stored_dataset = {}
            #     self.PreCalculating()
            if self.mode == 'TRAIN':
                if not os.path.exists(self.opt.args['ExpDir']+'Trainset.pt'):
                    self.stored_dataset = {}
                    self.PreCalculating()
                    t.save(self.stored_dataset, self.opt.args['ExpDir']+'Trainset.pt')
                else:
                    self.stored_dataset = t.load(self.opt.args['ExpDir']+'Trainset.pt')
            else:
                self.stored_dataset = {}
                self.PreCalculating()


        if opt.args['Feature'] == 'SMILES':
            print('Prefeaturizing...')
            self.featurizer.prefeaturize(self.dataset)

    def BuildFeaturizer(self):
        feature_choice = self.opt.args['Feature']
        if feature_choice == 'FP':
            self.featurizer = FPFeaturizer(self.opt)
        elif feature_choice == 'Graph':
            self.featurizer = GraphFeaturizer()
        elif feature_choice == 'AttentiveFP':
            self.featurizer = AttentiveFPFeaturizer(
                    atom_feature_size = self.opt.args['AtomFeatureSize'],
                    bond_feature_size = self.opt.args['BondFeatureSize'],
                    max_degree = 5
            )
        elif feature_choice == 'FraGAT':
            self.featurizer = FraGATFeaturizer(
            atom_feature_size = self.opt.args['AtomFeatureSize'],
            bond_feature_size = self.opt.args['BondFeatureSize'],
            max_degree = 5,
            mode = self.mode
            )
        elif feature_choice == 'PyGFraGAT':
            self.featurizer = PyGFraGATFeaturizer(self.opt, self.mode)
        elif feature_choice == 'CMPNN':
            self.featurizer = CMPNNFeaturizer(self.opt)
        elif feature_choice == 'Graphormer':
            self.featurizer = GraphormerFeaturizer(
                    max_node = self.opt.args['max_node'],
                    multi_hop_max_dist = self.opt.args['multi_hop_max_dist'],
                    spatial_pos_max = self.opt.args['spatial_pos_max'],
                    opt = self.opt
            )
        elif feature_choice == 'SMILES':
            self.featurizer = SMILESTokenFeaturizer(self.opt)
        # todo(zqzhang): updated in FraContra
        elif feature_choice == 'FraContra':
            self.featurizer = PyGFraGATContraFeaturizer(self.opt)
        elif feature_choice == 'FraContra-finetune':
            self.featurizer = PyGFraGATContraFinetuneFeaturizer(self.opt)
            # print(f"Apoint")
            # self.featurizer = PyGFraGATContraFeaturizer(self.opt)



        else:
            raise KeyError("Wrong feature option!")

    def __getitem__(self, index):
        if (self.featurizer.__class__ == AttentiveFPFeaturizer) or (self.featurizer.__class__ == FraGATFeaturizer):
            value = self.dataset[index]['Value']
            smiles = self.dataset[index]['SMILES']
            idx = t.Tensor([self.dataset[index]['idx']]).long()
            mol = GetMol(smiles)
            data, label = self.featurizer.featurize(self.prefeaturized_dataset, index, mol, value, self.opt)
        elif (self.featurizer.__class__ == GraphormerFeaturizer):
            item = self.dataset[index]
            idx = t.Tensor([self.dataset[index]['idx']]).long()
            data= self.featurizer.featurize(item)
            data.append(idx)
            return data
        elif (self.featurizer.__class__ == PyGFraGATFeaturizer):
            item = self.dataset[index]
            if self.mode == 'EVAL':
                if index in self.stored_dataset.keys():
                    return self.stored_dataset[index]
            idx = t.Tensor([self.dataset[index]['idx']]).long()
            data = self.featurizer.featurize(item)
            data.__setattr__('idx', idx)
            if self.mode == 'Eval':
                self.stored_dataset.update({index:data})
            return data

        # todo(zqzhang): updated in FraContra
        elif (self.featurizer.__class__ == PyGFraGATContraFeaturizer):
            item = self.dataset[index]
            if self.mode == 'TRAIN':
                if self.opt.args['SplittingTrainset']:
                    if len(self.stored_dataset)==0:
                        if os.path.exists(self.opt.args[
                                              'PretrainDataPath'] + f"_PreCalculatedTensors_seed{self.opt.args['PretrainDatasetValidSeed']}_{self.opt.args['PretrainValidSplitRate']}_Trainset_Part_{self.TrainsetPart}.pt"):
                            print(f"Loading part {self.TrainsetPart} of Trainset...")
                            self.stored_dataset = t.load(self.opt.args[
                                                             'PretrainDataPath'] + f"_PreCalculatedTensors_seed{self.opt.args['PretrainDatasetValidSeed']}_{self.opt.args['PretrainValidSplitRate']}_Trainset_Part_{self.TrainsetPart}.pt")

                    if index in self.stored_dataset.keys():
                        return self.stored_dataset[index]

            if self.mode == 'EVAL':
                if index in self.stored_dataset.keys():
                    return self.stored_dataset[index]
            idx = t.Tensor([index]).long()
            data_mol, data_frag, data_SMILES = self.featurizer.featurize(item)
            data_mol.__setattr__('idx',idx)
            return (data_mol, data_frag, data_SMILES)

        elif (self.featurizer.__class__ == PyGFraGATContraFinetuneFeaturizer):
            item = self.dataset[index]
            # if self.mode == 'EVAL':
            #     if index in self.stored_dataset.keys():
            #         return self.stored_dataset[index]
            if index in self.stored_dataset.keys():
                return self.stored_dataset[index]
            data = self.featurizer.featurize(item)
            # if self.mode == 'EVAL':
            #     self.stored_dataset.update({index:data})
            self.stored_dataset.update({index:data})
            return data
        else:
            item = self.dataset[index]
            idx = t.Tensor([self.dataset[index]['idx']]).long()
            data, label = self.featurizer.featurize(item)
        return data, label, idx

    def PreCalculating(self):
        for idx in tqdm(range(len(self.dataset))):
            # if idx % (len(self.dataset)/10) == 0:
            #     print(f"{idx / len(self.dataset)} calculated...")
            data = self.__getitem__(idx)
            self.stored_dataset.update({idx:data})

    def __len__(self):
        return len(self.dataset)

class PyGMolDataset(InMemoryDataset):
    def __init__(self, graphdataset, opt, mode):
        self.graph_dataset = graphdataset
        self.opt = opt
        # todo(zqzhang): updated in ACv7
        self.dataset_path_root = self.opt.args['ExpDir'] + 'Dataset/'
        if not os.path.exists(self.dataset_path_root):
            os.mkdir(self.dataset_path_root)
        self.mode = mode
        if os.path.exists(self.dataset_path_root + 'processed/' + self.processed_file_names[0]):
            os.remove(self.dataset_path_root + 'processed/' + self.processed_file_names[0])
        super(PyGMolDataset, self).__init__(root = self.dataset_path_root)
        self.data, self.slices = t.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.opt.args['DataPath']]

    @property
    def processed_file_names(self):
        return [self.opt.args['ExpName'] + '_' + self.mode + '.pt']

    def download(self):
        pass

    def process(self):
        data_list = self.graph_dataset
        data, slices = self.collate(data_list)
        # print("Processed without saving complete.")
        print("Saving processed files...")
        t.save((data, slices), self.processed_paths[0])
        print('Saving complete!')



    # def __len__(self):
    #     return len(self.graph_dataset)

# todo(zqzhang): updated in TPv7
class PretrainedMolDataset(data.Dataset):
    # Mol Dataset Module for Pretrain-finetune experiment
    def __init__(self, dataset, opt, mode):
        super(PretrainedMolDataset, self).__init__()
        self.dataset = dataset
        self.opt = opt
        self.featurizer = PretrainFeatureFeaturizer(self.opt)

    def __getitem__(self, index):
        item = self.dataset[index]
        idx = t.Tensor([item['idx']]).long()
        data, label = self.featurizer.featurize(item)
        return data,label,idx


    def __len__(self):
        return len(self.dataset)

# todo(zqzhang): updated in TPv7
class ToysetDataset(data.Dataset):
    def __init__(self, dataset, opt, mode):
        super(ToysetDataset, self).__init__()
        self.dataset = dataset
        self.opt = opt
        if not self.opt.args['PyG']:
            if self.opt.args['Feature'] == 'Raw':
                self.featurizer = RawFeatureFeaturizer(self.opt)
        else:
            self.featurizer = PyGGraphFeaturizer(self.opt)

    def __getitem__(self, index):
        item = self.dataset[index]
        idx = t.Tensor([item['idx']]).long()
        data, label = self.featurizer.featurize(item)
        # print(f"data: {data}")
        # print(f"label: {label}")
        # print(f"idx: {idx}")
        return data, label, idx

    def __len__(self):
        return len(self.dataset)
########################################################################################

#######################################################################################
class MolDatasetCreator(object):
    # An object to create molecule datasets from a given dataset file path.
    # Using CreateDatasets function to generate 2 or 3 datasets, based on the SplitRate
    # Based on the MolDatasetCreator above, this version added the MSN creating part
    # including the network building and the mask creating according to the splitting.
    def __init__(self, opt):
        super(MolDatasetCreator, self).__init__()
        self.FileParserList = {
            'HIV': HIVFileParser(),
            'BBBP': BBBPFileParser(),
            'Tox21': Tox21FileParser(),
            'FreeSolv': FreeSolvFileParser(),
            'ESOL': ESOLFileParser(),
            'QM9': QM9FileParser(),
            'BACE': BACEFileParser(),
            'ClinTox': ClinToxFileParser(),
            'SIDER': SIDERFileParser(),
            'SHP2': SHP2FileParser(),
            'Toxcast': ToxcastFileParser(),
            'Lipop': LipopFileParser(),
            'CEP': CEPFileParser(),
            'Malaria': MalariaFileParser(),
            'MUV': MUVFileParser()
        }
        self.SplitterList = {
            'Random': RandomSplitter(),
            'MultitaskRandom': MultitaskRandomSplitter(),
            'Scaffold': ScaffoldSplitter(),
            'ScaffoldRandom': ScaffoldRandomSplitter()
        }
        self.opt = opt

    def CalculateWeight(self, dataset):
        weights = []
        task_num = self.opt.args['TaskNum']
        for i in range(task_num):
            pos_count = 0
            neg_count = 0
            for item in dataset:
                value = item['Value'][i]
                if value == '0':
                    neg_count += 1
                elif value == '1':
                    pos_count += 1
            pos_weight = (pos_count + neg_count) / pos_count
            neg_weight = (pos_count + neg_count) / neg_count
            weights.append([neg_weight, pos_weight])
        return weights

    def CreateDatasets(self, dataset = None):
        if not dataset:
            # read the dataset, get raw_data
            file_path = self.opt.args['DataPath']
            print("Loading data file...")
            fileloader = FileLoader(file_path)
            raw_data = fileloader.load()

            # parse raw_data, get raw_dataset
            print("Parsing lines...")
            parser = self.FileParserList[self.opt.args['ExpName']]
            raw_dataset = parser.parse_file(raw_data)
            # raw_dataset is a list in type of : {'SMILES': , 'Value': ,}
            print("Dataset is parsed. Original size is ", len(raw_dataset))

            # raw_dataset is the original dataset from the files
            # before processing, checker should be used to screen the samples that not satisfy the rules.
            # if using AttentiveFP models, extra rules are used to check the dataset according to the Attentive FP.
            # otherwise, only MolChecker is used to make sure that all of the samples are validity to the rdkit.
            if (self.opt.args['Feature'] == 'AttentiveFP') or (self.opt.args['Feature'] == 'FraGAT') \
                or (self.opt.args['Feature'] == 'Graphormer') or (self.opt.args['Feature'] == 'FraContra-finetune'):
                print('Using checking rules proposed in Attentive FP. Dataset is being checked.')
                self.checker = AttentiveFPChecker(max_atom_num=102, max_degree=5)
                self.screened_dataset = self.checker.check(raw_dataset)
            else:
                # todo(zqzhang): updated in TPv7.8!
                if self.opt.args['PyG']:
                    print(f"Checking dataset...")
                    self.checker = AttentiveFPChecker(max_atom_num=102, max_degree=5)
                    self.screened_dataset = self.checker.check(raw_dataset)
                else:
                    self.checker = MolChecker()
                    self.screened_dataset = self.checker.check(raw_dataset)

            # The idx of each sample should be added after screening,
            # because the MSN is created based on the screened_dataset,
            # so that the idx of the samples is determined by the location of it in the screened_dataset.
            for idx, data in enumerate(self.screened_dataset):
                data.update({'idx': idx})

            self.CheckScreenedDatasetIdx()
        else:
            self.screened_dataset = dataset

        # if QM9 dataset, values need to be normalized
        # todo(zqzhang): updated in TPv8
        if self.opt.args['ExpName'] == 'QM9':
            normalized_dataset_filename = self.opt.args['TrialPath'] + 'NormalizedDataset.json'
            if not os.path.exists(normalized_dataset_filename):
                print(f"Normalizing Values of QM9")
                self.screened_dataset = self.ValueNormalization(self.screened_dataset)
                # print(self.screened_dataset[0])
            else:
                print(f"Loading Normalized dataset of QM9")
                with open(normalized_dataset_filename, 'r') as f:
                    self.screened_dataset = json.load(f)
                # print(self.screened_dataset[0])
            # raise RuntimeError

        # Check for debug
        # print(f"55179: {self.screened_dataset[55179]}")
        # print(f"SMILES: {self.screened_dataset[55179]['SMILES']}")
        # raise RuntimeError

        # After checking, all of the "raw_dataset" should be taken place by "self.screened_dataset"
        if self.opt.args['ClassNum'] == 2:  # only binary classification tasks needs to calculate weights.
            if self.opt.args['Weight']:
                weights = self.CalculateWeight(self.screened_dataset)
            else:
                weights = None
        else:
            weights = None
        # weights is a list with length of 'TaskNum'.
        # It shows the distribution of pos/neg samples in the dataset(Screened dataset, before splitting and after screening)
        # And it is used as a parameter of the loss function to balance the learning process.
        # For multitasks, it contains multiple weights.

        if self.opt.args['Splitter']:
            splitter = self.SplitterList[self.opt.args['Splitter']]
            print("Splitting dataset...")
            sets, idxs = splitter.split(self.screened_dataset, self.opt)

            if len(sets) == 2:
                trainset, validset = sets
                self.trainidxs, self.valididxs = idxs
                print("Dataset is splitted into trainset: ", len(trainset), " and validset: ", len(validset))

            if len(sets) == 3:
                trainset, validset, testset = sets
                self.trainidxs, self.valididxs, self.testidxs = idxs
                print("Dataset is splitted into trainset: ", len(trainset), ", validset: ", len(validset),
                      " and testset: ", len(testset))
        else:
            trainset = self.screened_dataset
            sets = (trainset)


        # Construct dataset objects of subsets
        if not self.opt.args['PyG']:
            Trainset = MolDataset(trainset, self.opt, 'TRAIN')
            if len(sets) == 2:
                Validset = MolDataset(validset, self.opt, 'EVAL')
                return (Trainset, Validset), weights
            elif len(sets) == 3:
                Validset = MolDataset(validset, self.opt, 'EVAL')
                Testset = MolDataset(testset, self.opt, 'EVAL')
                return (Trainset, Validset, Testset), weights
            else:
                return (Trainset), weights
        else:
            if self.opt.args['Feature'] == 'PyGFraGAT':
                Trainset = MolDataset(trainset, self.opt, 'TRAIN')
                Validset = MolDataset(validset, self.opt, 'EVAL')
                if len(sets) == 3:
                    Testset = MolDataset(testset, self.opt, 'EVAL')
                    return (Trainset, Validset, Testset), weights
                else:
                    # Testset = None
                    return (Trainset, Validset), weights
                # PyGTrainset = PyGMolDataset(Trainset, self.opt, 'TRAIN')
                # PyGValidset = PyGMolDataset(Validset, self.opt, 'EVAL')
                # if Testset:
                #     PyGTestset = PyGMolDataset(Testset, self.opt, 'EVAL')
                #     return (PyGTrainset, PyGValidset, PyGTestset), weights
                # else:
                #     return (PyGTrainset, PyGValidset), weights


            else:
                pyg_feater = PyGGraphFeaturizer(self.opt)
                PyGGraphTrainset = []
                for sample in trainset:
                    graph_sample = pyg_feater.featurize(sample)
                    PyGGraphTrainset.append(graph_sample)
                # print(PyGGraphTrainset)
                # print(len(PyGGraphTrainset))
                Trainset = PyGMolDataset(PyGGraphTrainset, self.opt, 'TRAIN')
                # print(Trainset)
                # print(len(Trainset))
                if len(sets) == 2:
                    PyGGraphValidset = []
                    for sample in validset:
                        graph_sample = pyg_feater.featurize(sample)
                        PyGGraphValidset.append(graph_sample)
                    Validset = PyGMolDataset(PyGGraphValidset, self.opt, 'VALID')
                    return (Trainset, Validset), weights
                elif len(sets) == 3:
                    PyGGraphValidset = []
                    for sample in validset:
                        graph_sample = pyg_feater.featurize(sample)
                        PyGGraphValidset.append(graph_sample)
                    Validset = PyGMolDataset(PyGGraphValidset, self.opt, 'VALID')
                    PyGGraphTestset = []
                    for sample in testset:
                        graph_sample = pyg_feater.featurize(sample)
                        PyGGraphTestset.append(graph_sample)
                    Testset = PyGMolDataset(PyGGraphTestset, self.opt, 'TEST')
                    return (Trainset, Validset, Testset), weights

    # todo(zqzhang): updated in TPv8
    def ValueNormalization(self, dataset):
        import numpy as np
        from TrainingFramework.FileUtils import ZscoreNormalization
        Labels = []
        for i in range(len(dataset)):
            item = dataset[i]
            label = []
            values = item['Value']
            for v in values:
                label.append(float(v))
            Labels.append(label)
        Labels = np.array(Labels)
        assert Labels.shape[1] == 12
        normalized_Labels, z_mean, z_var = ZscoreNormalization(Labels)
        normalized_dataset = []
        for i in range(len(dataset)):
            item = dataset[i]
            norm_values = normalized_Labels[i]
            assert (float(item['Value'][3]) - z_mean[3]) / np.sqrt(z_var[3]) == norm_values[3]
            values = []
            for v in norm_values:
                values.append(str(v))
            normalized_dataset.append({'SMILES':item['SMILES'],'Value':values, 'idx': item['idx']})

        filename = self.opt.args['TrialPath']
        with open(filename+'NormalizedDataset.json', 'w') as f:
            json.dump(normalized_dataset, f)
        np.save(filename+'ValueMean.npy', z_mean)
        np.save(filename+'ValueVar.npy',z_var)
        return normalized_dataset






    def CheckScreenedDatasetIdx(self):
        print("Check whether idx is correct: ")
        chosen_idx = int(random.random() * len(self.screened_dataset))
        print(chosen_idx)
        print(self.screened_dataset[chosen_idx])
        assert chosen_idx == self.screened_dataset[chosen_idx]['idx']

    def GenerateMSNMasks(self, MSN):
        total_num = MSN.NodeNum()
        if len(self.opt.args['SplitRate']) == 2:
            train_mask = np.zeros(total_num)
            val_mask = np.zeros(total_num)
            test_mask = np.zeros(total_num)
            train_mask[self.trainidxs] = 1
            val_mask[self.valididxs] = 1
            test_mask[self.testidxs] = 1
            train_mask = t.Tensor(train_mask)
            val_mask = t.Tensor(val_mask)
            test_mask = t.Tensor(test_mask)
            MSN.WriteFeaturesToDGLGraph('train_mask', train_mask, 'mask')
            MSN.WriteFeaturesToDGLGraph('val_mask', val_mask, 'mask')
            MSN.WriteFeaturesToDGLGraph('test_mask', test_mask, 'mask')
        elif len(self.opt.args['SplitRate']) == 1:
            train_mask = np.zeros(total_num)
            val_mask = np.zeros(total_num)
            train_mask[self.trainidxs] = 1
            val_mask[self.valididxs] = 1
            train_mask = t.Tensor(train_mask)
            val_mask = t.Tensor(val_mask)
            MSN.WriteFeaturesToDGLGraph('train_mask', train_mask, 'mask')
            MSN.WriteFeaturesToDGLGraph('val_mask', val_mask, 'mask')
        else:
            train_mask = np.zeros(total_num)
            train_mask[self.trainidxs] = 1
            train_mask = t.Tensor(train_mask)#.bool()
            MSN.WriteFeaturesToDGLGraph('train_mask', train_mask, 'mask')

    def ValidDatasetAndMasks(self, sets, masks, values):
        (trainset, validset, testset) = sets
        (trainmask, validmask, testmask) = masks
        for item in trainset:
            idx = item['idx']
            smiles = item['SMILES']
            value = item['Value']
            original_smiles = self.screened_dataset[idx]['SMILES']
            original_value = self.screened_dataset[idx]['Value']
            assert smiles == original_smiles
            assert value == original_value
            assert trainmask[idx] == True
            assert validmask[idx] == False
            assert testmask[idx] == False
            assert values[idx] == float(value)
        for item in validset:
            idx = item['idx']
            smiles = item['SMILES']
            value = item['Value']
            original_smiles = self.screened_dataset[idx]['SMILES']
            original_value = self.screened_dataset[idx]['Value']
            assert smiles == original_smiles
            assert value == original_value
            assert trainmask[idx] == False
            assert validmask[idx] == True
            assert testmask[idx] == False
            assert values[idx] == float(value)
        for item in testset:
            idx = item['idx']
            smiles = item['SMILES']
            value = item['Value']
            original_smiles = self.screened_dataset[idx]['SMILES']
            original_value = self.screened_dataset[idx]['Value']
            assert smiles == original_smiles
            assert value == original_value
            assert trainmask[idx] == False
            assert validmask[idx] == False
            assert testmask[idx] == True
            assert values[idx] == float(value)

# todo(zqzhang): updated in TPv7
class PretrainMolDatasetCreator(object):
    def __init__(self, opt):
        super(PretrainMolDatasetCreator, self).__init__()
        self.opt = opt
        self.SplitterList = {
            'Random': RandomSplitter(),
            'MultitaskRandom': MultitaskRandomSplitter(),
            'ScaffoldRandom': ScaffoldRandomSplitter()
        }   # only Random Splitter and MultitaskRandom is available.

    def CalculateWeight(self, dataset):
        weights = []
        task_num = self.opt.args['TaskNum']
        for i in range(task_num):
            pos_count = 0
            neg_count = 0
            for item in dataset:
                value = item['Value'][i]
                if value == 0:
                    neg_count += 1
                elif value == 1:
                    pos_count += 1
            pos_weight = (pos_count + neg_count) / pos_count
            neg_weight = (pos_count + neg_count) / neg_count
            weights.append([neg_weight, pos_weight])
        return weights

    def CreateDatasets(self, dataset=None):
        if not dataset:
            # Read the Pretrain-TargetTask Feature Datasets.
            feature_file_path = self.opt.args['FeaturePath']
            print("Loading Pretrain-TargetTask Feature dataset...")
            fileloader = PTFileLoader(feature_file_path)
            raw_feature_dataset = fileloader.load()
            # raw_feature_dataset: [total_num, feature_length]

            value_file_path = self.opt.args['ValuePath']
            fileloader = NpyFileLoader(value_file_path)
            raw_value_dataset = fileloader.load()
            raw_value_dataset = raw_value_dataset.T
            # raw_label_dataset: [total_num, task_num]

            assert len(raw_feature_dataset) == len(raw_value_dataset)

            # Construct the {'Feature':,'Value':, 'idx':} dataset structure.
            total_num = len(raw_feature_dataset)
            dataset = []
            for i in range(total_num):
                feature = raw_feature_dataset[i]
                value = raw_value_dataset[i]
                item = {'Feature': feature, 'Value': value, 'idx': i}
                # Feature type: t.Tensor
                # Value type: np.ndarray
                # Value shape: [task_num]
                dataset.append(item)

            self.screened_dataset = dataset
        else:
            self.screened_dataset = dataset

        if self.opt.args['ClassNum'] == 2:  # only binary classification tasks needs to calculate weights.
            if self.opt.args['Weight']:
                weights = self.CalculateWeight(self.screened_dataset)
            else:
                weights = None
        else:
            weights = None

        if self.opt.args['Splitter']:
            splitter = self.SplitterList[self.opt.args['Splitter']]
            print("Splitting dataset...")
            sets, idxs = splitter.split(self.screened_dataset, self.opt)

            if len(sets) == 2:
                trainset, validset = sets
                self.trainidxs, self.valididxs = idxs
                print("Dataset is splitted into trainset: ", len(trainset), " and validset: ", len(validset))

            if len(sets) == 3:
                trainset, validset, testset = sets
                self.trainidxs, self.valididxs, self.testidxs = idxs
                print("Dataset is splitted into trainset: ", len(trainset), ", validset: ", len(validset),
                      " and testset: ", len(testset))
        else:
            trainset = self.screened_dataset
            sets = (trainset)

        Trainset = PretrainedMolDataset(trainset, self.opt, 'TRAIN')
        if len(sets) == 2:
            Validset = PretrainedMolDataset(validset, self.opt, 'EVAL')
            return (Trainset, Validset), weights
        elif len(sets) == 3:
            Validset = PretrainedMolDataset(validset, self.opt, 'EVAL')
            Testset = PretrainedMolDataset(testset, self.opt, 'EVAL')
            return (Trainset, Validset, Testset), weights
        else:
            return (Trainset), weights

# todo(zqzhang): updated in TPv7
class ToysetCreator(object):
    def __init__(self, opt):
        super(ToysetCreator, self).__init__()
        self.opt = opt
        self.splitter = RandomSplitter()

    def CalculateWeight(self, dataset):
        weights = []
        # task_num = self.opt.args['TaskNum']
        pos_count = 0
        neg_count = 0
        for item in dataset:
            value = item['Value']
            if value == 0:
                neg_count+=1
            elif value == 1:
                pos_count += 1
        pos_weight = (pos_count + neg_count) / pos_count
        neg_weight = (pos_count + neg_count) / neg_count
        weights.append([neg_weight, pos_weight])
        return weights

    def CreateDatasets(self, dataset=None):
        if not dataset:
            data_path = self.opt.args['DataPath']
            suffix = re.split('\.', data_path)[-1]
            if suffix == 'json':
                fileloader = JsonFileLoader(data_path)
                self.screened_dataset = fileloader.load()
            elif suffix == 'pt':
                fileloader = PTFileLoader(data_path)
                self.screened_dataset = fileloader.load()
            else:
                raise RuntimeError("Trying to load an unknown suffix file.")

            if 'idx' not in self.screened_dataset[0].keys():
                for idx, data in enumerate(self.screened_dataset):
                    data.update({'idx': idx})
        else:
            self.screened_dataset = dataset

            if 'idx' not in self.screened_dataset[0].keys():
                for idx, data in enumerate(self.screened_dataset):
                    data.update({'idx': idx})

        if self.opt.args['ClassNum'] == 2:  # only binary classification tasks needs to calculate weights.
            if self.opt.args['Weight']:
                weights = self.CalculateWeight(self.screened_dataset)
            else:
                weights = None
        else:
            weights = None

        if self.opt.args['Splitter']:
            print("Splitting dataset...")
            sets, idxs = self.splitter.split(self.screened_dataset, self.opt)

            if len(sets) == 2:
                trainset, validset = sets
                self.trainidxs, self.valididxs = idxs
                print("Dataset is splitted into trainset: ", len(trainset), " and validset: ", len(validset))

            if len(sets) == 3:
                trainset, validset, testset = sets
                self.trainidxs, self.valididxs, self.testidxs = idxs
                print("Dataset is splitted into trainset: ", len(trainset), ", validset: ", len(validset),
                      " and testset: ", len(testset))
        else:
            trainset = self.screened_dataset
            sets = (trainset)

        Trainset = ToysetDataset(trainset, self.opt, 'TRAIN')
        if len(sets) == 2:
            Validset = ToysetDataset(validset, self.opt, 'EVAL')
            return (Trainset, Validset), weights
        elif len(sets) == 3:
            Validset = ToysetDataset(validset, self.opt, 'EVAL')
            Testset = ToysetDataset(testset, self.opt, 'EVAL')
            return (Trainset, Validset, Testset), weights
        else:
            return (Trainset), weights




#######################################################################################
def collate_first_dim(batch):
    all_batch_tensor, labels = map(list, zip(*batch))
    list_res_each_position = [[] for i in range(16)]
    len_each_position = [[] for i in range(16)]
    for one_16 in all_batch_tensor:
        for index, one_tensor in enumerate(one_16):
            list_res_each_position[index].append(one_tensor)
            len_each_position[index].append(len(one_tensor))
    len_each_position = len_each_position[0]
    res = []
    for item_list in list_res_each_position:
        one_res = t.stack(item_list, dim = 0)
        res.append(one_res)
    labels = t.tensor(labels).long()
    return res, labels, len_each_position

###################################################################################
# Test codes
#

