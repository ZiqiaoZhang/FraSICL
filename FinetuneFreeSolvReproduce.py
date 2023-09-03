# from TrainingFramework.ProcessControllers import *
from ContrastiveComponent.ContrastiveProcessController import *

import os
os.environ['CUDA_LAUNCH_BLOCKING']='1'

ExpOptions = {
    'Search': 'greedy',
    'SeedPerOpt': 1,
    'SeedSearch': True,
    'PretrainFinetune':'Finetune',
    'TorchSeedBias': 21,
}

BasicParamList = {
    'ExpName': 'FreeSolv',
    'PretrainDataPath':'/remote-home/zqzhang/Data/PretrainData/pubchem-200K-screened.txt',
    'PretrainedModelPath':'/remote-home/zqzhang/Experiments/FraContra/PretrainedModels/ModelForPublish/model-58-fin',
    'MainMetric': 'RMSE',
    'DataPath': './Datasets/DrugData/FreeSolv_SMILESValue.txt',
    'RootPath': '/remote-home/zqzhang/Experiments/FraContra/FinetuneSetting3/Model11/Reproduce/',
    'CUDA_VISIBLE_DEVICES': '0',
    'TaskNum': 1,
    'ClassNum': 1,
    'OutputSize': 1,
    'Feature': 'FraSICL-finetune',
    'Model': 'FraSICL',

    'OnlySpecific': False,
    'Weight': True,
    'AC': False,
    'PyG': False,

    'FinetunePTM': True,
    'UseWhichView':'Frag',

    'ValidRate': 4000,
    'PrintRate': 2,
    'UpdateRate': 1,
    'SaveCkptRate':200,
    'PretrainValidSplitRate':0.95,
    'ValidBalance': False,
    'TestBalance': False,
    'SplitRate': [0.8, 0.1],
    'Splitter': 'ScaffoldRandom',
    'MaxEpoch': 300,
    'LowerThanMaxLimit': 30,
    'DecreasingLimit': 12,

    # if OnlyEval == True:
    'EvalModelPath': None,
    'EvalDatasetPath': None,
    # todo(zqzhang): updated in TPv7
    'EvalBothSubsets': None,
    'EvalLogAllPreds': False,            # LogAllPreds can only be used when EvalBothSubsets==False
    'EvalOptPath': None,
    ################################

    'Scheduler': 'PolynomialDecayLR',

    # Params for PolynomialDecayLR only
    'WarmupEpoch': 2,
    'LRMaxEpoch':300,
    'EndLR':1e-9,
    'Power':1.0,
    # Params for StepLR only
    'LRStep': 30,
    'LRGamma': 0.1,
    #####################################

    'WeightIniter': None,            #Choice: Norm, XavierNorm

    # Params for NormWeightIniter only
    'InitMean' : 0,
    'InitStd' : 1,

    #####################################

    # todo(zqzhang): updated in GBv1
    'FeatureCategory': 'BaseOH',
    'AtomFeatureSize': 39,
    'BondFeatureSize': 10,
    # 'Frag': False,
    'MolFP': 'MorganFP',
    'radius': 2,
    'nBits': 1024,

    # Training Params to be adujsted. If the param is not needed to be adjusted, set the value here.
    # 'SplitValidSeed': 8,
    # 'SplitTestSeed': 8,
    # 'FPSize': 32,
    # 'lr': None,
    # 'DropRate': None,
    # 'WeightDecay': None,
    # 'BatchSize': 200,
    # 'DNNLayers': [],

    # Params for FraSICL
    'BackbonePretrainModel-mol': 'PyGATFP',
    'BackbonePretrainModel-frag': 'PyGATFP',
    'TransformerHiddenSize':256,
    'TransformerFFNSize':256,
    'TransformerNumHeads':30,
    'TransformerEncoderLayers':4,
    'FPSize':300,
    'ProjectHeadSize':300,
    'DropRate':0.2,
    'lr':3,
    'L2LossGamma':0.01,
    'NTXentLossTemp':0.1,
    'WeightDecay':5,
    'BatchSize':128,
    'ATFPInputSize_mol':300,
    'ATFPHiddenSize_mol':300,
    'AtomLayers_mol':4,
    'MolLayers_mol':2,

    'ATFPInputSize_frag':300,
    'ATFPHiddenSize_frag':300,
    'AtomLayers_frag':5,
    'MolLayers_frag':2,


}

AdjustableParamList = {

}
SpecificParamList = {
    'SplitValidSeed': [588],
    'SplitTestSeed': [8],
    'DNNLayers': [[]],
    'lr': [4],
    'DropRate':[0.2],
    'PTMlr': [3],
    'BatchSize': [200],
}


expcontroller = PretrainExperimentProcessController(ExpOptions, [BasicParamList, AdjustableParamList, SpecificParamList])

expcontroller.ExperimentStart()

