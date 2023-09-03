import json
import os
import numpy as np
# todo(zqzhang): updated in TPv7
import re
import time
import random
# todo(zqzhang): updated in TPv7
from TrainingFramework.Dataset import MolDatasetCreator, MolDataset, PretrainMolDatasetCreator, PretrainedMolDataset, ToysetCreator
from TrainingFramework.FileUtils import JsonFileLoader
from TrainingFramework.Evaluator import *
from TrainingFramework.Scheduler import *
from TrainingFramework.Initializer import *

import torch.optim as optim
from functools import partial

from Models.FraGAT.FraGATModel import *
from Models.CMPNN.CMPNNModel import *
from Models.BasicGNNs import *
from Models.BasicLMs import *
from Models.Graphormer.Graphormer import *
from Models.Graphormer.collator import collator

# from ContrastiveComponent.ContrastiveModel import FraContra

class Saver(object):
    # Module that package the saving functions
    def __init__(self):
        super(Saver, self).__init__()
        #self.ckpt_state = {}

    def SaveContext(self, context_add, context_obj):
        # if something can be summarized as a dict {}
        # then using SaveContext function to save it into a json file.
        with open(context_add, 'w') as f:
            json.dump(context_obj, f)

    def LoadContext(self, context_add):
        # Using LoadContext to load a json file to a dict {}.
        with open(context_add, 'r') as f:
            obj = json.load(f)
        return obj

class Configs(object):
    def __init__(self, ParamList):
        # initiale a Config object with given paramlist
        super(Configs, self).__init__()
        self.args = {}
        for param in ParamList.keys():
            self.set_args(param, ParamList.get(param))

    def set_args(self, argname, value):
        if argname in self.args:
            print("Arg", argname, "is updated.")
            self.args[argname] = value
        else:
            print('Arg', argname, 'is added.')
            self.args.update({argname: value})

##########################################################################################################

class Controller(object):
    # Controllers are modules to control the entire experiment progress
    # A controller will maintain some values and params and make decisions based on the given results.
    # The values and params maintained by the controller are called running status.
    # Controllers should be able to save and load the running status to/from a json file so that the experimental progress
    # can be continued after accidental terminating.
    def __init__(self):
        super(Controller, self).__init__()

    def BuildControllerStatus(self):
        raise NotImplementedError(
            "Build Status function not implemented.")

    def GetControllerStatus(self):
        raise NotImplementedError(
            "Get Status function not implemented.")

    def SetControllerStatus(self, status):
        raise NotImplementedError(
            "Set Status function not implemented.")

    def AddControllerStatus(self, name, value):
        raise NotImplementedError(
            "Add Status function not implemented.")

class ControllerStatusSaver(object):
    # Package functions for saving and loading status of a controller into/from a file.
    # In a ControllerStatusSaver, it maintains three global variable:
    # self.args: args
    # self.saver: Saver() object for file saving and loading.
    # self.Addr: The Addr to save the status of the controller.

    def __init__(self, args, ControllerType, Addr=None, restart=False):
        super(ControllerStatusSaver, self).__init__()
        self.saver = Saver()
        self.args = args

        if ControllerType == 'ExperimentProcessController':
            self.Addr = self.args['TrialPath'] + 'ExperimentProcessControllerStatus/'
        elif ControllerType == 'ConfigController':
            self.Addr = self.args['TrialPath'] + 'ConfigControllerStatus/'
        elif ControllerType == 'EarlyStopController':
            self.Addr = self.args['SaveDir'] + 'EarlyStopControllerStatus/'
        elif ControllerType == 'Trainer':
            self.Addr = self.args['SaveDir'] + 'TrainerStatus/'
        elif ControllerType == 'CkptController':
            self.Addr = self.args['TrialPath'] + 'ConfigControllerStatus/'
        else:
            if Addr:
                self.Addr = Addr
            else:
                raise KeyError(
                        'Wrong ControllerType given.'
                )
        self.CheckAddr()

        if restart:
            self.DeleteFilesInDir(self.Addr)

    def DeleteFilesInDir(self, addr):
        del_list = os.listdir(addr)
        for f in del_list:
            file_addr = addr + f
            os.remove(file_addr)

    def CheckAddr(self):
        if not os.path.exists(self.Addr):
            os.mkdir(self.Addr)

    def SaveStatus(self, status, restart=False):

        next_cnt = self.CountFileNames(self.Addr)
        if next_cnt != 0:
            assert self.LastFileName(self.Addr) == str(next_cnt-1)
            file_name = self.Addr + str(next_cnt)
        else:
            file_name = self.Addr + '0'
        self.saver.SaveContext(file_name, status)


    def LoadStatus(self, status_idx=None):
        # if the index is not given, then find the last file from the folder. the last file is the file to be loaded.
        # otherwise, the file of the given index is to be loaded.
        if not status_idx:
            file_name = self.Addr + self.LastFileName(self.Addr)
        else:
            file_name = self.Addr + str(status_idx)

        # if no file is to be loaded, then return None.
        # (e.g. empty in the folder or the given index not exists)
        if os.path.exists(file_name):
            return self.saver.LoadContext(file_name)
        else:
            return None

    def LastFileName(self, Addr):
        dir_files = os.listdir(Addr)
        # os.listdir returns the file names in Addr, only the names, without the Addr path.
        if dir_files:
            dir_files = sorted(dir_files, key=lambda x: os.path.getctime(os.path.join(Addr, x)))
            last_file = dir_files[-1]
        else:
            last_file = ' '
        return last_file

    def CountFileNames(self, Addr):
        dir_files = os.listdir(Addr)
        return len(dir_files)

def TestCodesForControllerStatusSaver():
    args = {'TrialPath': './TestExps/test/',
            'SaveDir': './TestExps/test/expi/'}
    controllerstatussaver = ControllerStatusSaver(args, 'ExperimentProcessController')
    status = controllerstatussaver.LoadStatus()
    print(status)
    status = {'1':1, '2':2}
    controllerstatussaver.SaveStatus(status)
    status = {'1':2, '2':3}
    controllerstatussaver.SaveStatus(status)
    status = {'1': 20, '2': 30}
    controllerstatussaver.SaveStatus(status)
    status = controllerstatussaver.LoadStatus()
    print(status)
    status = controllerstatussaver.LoadStatus(1)
    print(status)
    status = controllerstatussaver.LoadStatus(5)
    print(status)

##########################################################################################################

class EarlyStopController(Controller):
    # A module used to control the early stop part of the experimental progress.
    # It maintains the result of each epoch, max results, count of worse results
    # and to make decision whether the training progress should be early stopped.
    def __init__(self, opt):
        super(EarlyStopController, self).__init__()
        self.opt = opt
        # params coming from the opt are constant during the training progress of THIS opt
        self.MetricName = opt.args['MainMetric']
        self.LowerThanMaxLimit = opt.args['LowerThanMaxLimit']
        self.DecreasingLimit = opt.args['DecreasingLimit']
        # Other params are the running status of the EarlyStopController that should be saved and loaded by files.
        # initial MaxResult
        if self.opt.args['ClassNum'] == 1:
            self.MaxResult = 9e8
        else:
            self.MaxResult = 0
        # todo(zqzhang): updated in TPv7
        self.MaxResultModelIdx = 0
        self.LastResult = 0
        self.LowerThanMaxNum = 0
        self.DecreasingNum = 0
        self.TestResult = []

    def ShouldStop(self, score, ckpt_idx, testscore=None):
        # Make decision whether the training progress should be stopped.
        # When the current result is better than the MaxResult, then update thre MaxResult.
        # When the current result is worse that the MaxResult, then start to count.
        # When the num of epochs that the result is worse than the MaxResult exceed the LowerThanMaxLimit threshold, then stop
        # And when the result is persistently getting worse for over DecreasingLimit epochs, then stop.

        # score is the current Validation Result
        # ckpt_idx is the ckpt index
        # testscore is the result of the current model on the test set.

        MainScore = score[self.MetricName]
        if testscore:
            MainTestScore = testscore[self.MetricName]
        else:
            MainTestScore = None
        self.TestResult.append(MainTestScore)

        if self.opt.args['ClassNum'] != 1:
            # Classification, the larger the better
            if MainScore > self.MaxResult:
                self.MaxResult = MainScore
                self.MaxResultModelIdx = ckpt_idx
                self.LowerThanMaxNum = 0
                self.DecreasingNum = 0
                # all counts reset to 0.
            else:
                # decreasing, start to count.
                self.LowerThanMaxNum += 1
                if MainScore < self.LastResult:
                # decreasing consistently.
                    self.DecreasingNum += 1
                else:
                    self.DecreasingNum = 0
            self.LastResult = MainScore
        else:
            # Regression, the lower the better
            if MainScore < self.MaxResult:
                self.MaxResult = MainScore
                self.MaxResultModelIdx = ckpt_idx
                self.LowerThanMaxNum = 0
                self.DecreasingNum = 0
                # all set to 0.
            else:
                # decreasing, start to count.
                self.LowerThanMaxNum += 1
                if MainScore > self.LastResult:
                # decreasing consistently.
                    self.DecreasingNum += 1
                else:
                    self.DecreasingNum = 0
            self.LastResult = MainScore

        if self.LowerThanMaxNum > self.LowerThanMaxLimit:
            return True
        if self.DecreasingNum > self.DecreasingLimit:
            return True
        return False

    def BestModel(self):
        return self.MaxResult, self.MaxResultModelIdx, self.TestResult[self.MaxResultModelIdx]

    def GetControllerStatus(self):
        status = {
            'MaxResult': self.MaxResult,
            'MaxResultModelIdx': self.MaxResultModelIdx,
            'LastResult': self.LastResult,
            'LowerThanMaxNum': self.LowerThanMaxNum,
            'DecreasingNum': self.DecreasingNum,
            'TestResult': self.TestResult
        }
        return status

    def SetControllerStatus(self, status):
        self.MaxResult = status['MaxResult']
        self.MaxResultModelIdx = status['MaxResultModelIdx']
        self.LastResult = status['LastResult']
        self.LowerThanMaxNum = status['LowerThanMaxNum']
        self.DecreasingNum = status['DecreasingNum']
        self.TestResult = status['TestResult']

##########################################################################################################

class ConfigController(Controller):
    # A module to control the Configs of the training progress.
    # Including the configs for training, and the hyperparameters that should be searched.
    # The HyperParam Searching Methods can be modified.
    def __init__(self):
        super(ConfigController, self).__init__()

    def AdjustParams(self):
        raise NotImplementedError(
            "Adjust Params function not implemented.")

    def GetOpts(self):
        raise NotImplementedError(
            "Get Opts function not implemented.")

class GreedyConfigController(ConfigController):
    # Here the basic greedy searching strategy is implemented.

    def __init__(self, BasicHyperparamList, AdjustableHyperparamList, SpecificHyperparamList=None):
        # Basic: Configs for training, not for HyperParam Searching
        # Adjustable: Configs for greedy searching, candidates.
        # Specific: Specific group of HyperParams, not for greedy searching.
        super(ConfigController, self).__init__()
        self.BasicHyperparameterList = BasicHyperparamList
        self.HyperparameterList = AdjustableHyperparamList
        self.SpecificHyperparamList = SpecificHyperparamList
        self.opt = Configs(self.BasicHyperparameterList)
        self.MainMetric = self.BasicHyperparameterList['MainMetric']
        self.OnlySpecific = self.BasicHyperparameterList['OnlySpecific']

        # set the Trial Path for the experiment on this dataset.
        self.opt.set_args('TrialPath', self.opt.args['RootPath'] + self.opt.args['ExpName'] + '/')
        if not os.path.exists(self.opt.args['TrialPath']):
            os.mkdir(self.opt.args['TrialPath'])

        self.controllerstatussaver = ControllerStatusSaver(self.opt.args, 'ConfigController')
        status = self.LoadStatusFromFile()
        if status:
            self.SetControllerStatus(status)
        else:
            self.InitControllerStatus()
            self.CheckSpecificHyperparamList(SpecificHyperparamList)
            self.OptInit(SpecificHyperparamList)


    def CheckSpecificHyperparamList(self, SpecificHyperparamList):
        firstkey = list(SpecificHyperparamList.keys())[0]
        SpecificChoiceNum = len(SpecificHyperparamList[firstkey])
        for key in SpecificHyperparamList.keys():
            assert SpecificChoiceNum == len(SpecificHyperparamList[key])

    def OptInit(self, SpecificHyperparamList):
        if SpecificHyperparamList:
            self.HyperparameterInit(self.SpecificHyperparamList)
        else:
            self.HyperparameterInit(self.HyperparameterList)

    def HyperparameterInit(self, paramlist):
        for param in paramlist.keys():
            self.opt.set_args(param, paramlist.get(param)[0])
        # initially, the hyperparameters are set to be the first value of their candidate lists each.

    def GetOpts(self):
        self.opt.set_args('ExpDir', self.opt.args['TrialPath'] + 'exp' + str(self.exp_count) + '/')
        if not os.path.exists(self.opt.args['ExpDir']):
            os.mkdir(self.opt.args['ExpDir'])
        return self.opt

    def AdjustParams(self):
        # Adjust the hyperparameters by greedy search.
        # The return is the end flag

        # if the Specific Hyperparam List is given, then set the opts as the param group in SpecificParamList
        if self.SpecificHyperparamList:
            keys = self.SpecificHyperparamList.keys()
            if self.exp_count < len(self.SpecificHyperparamList.get(list(keys)[0])):
                for param in self.SpecificHyperparamList.keys():
                    self.opt.set_args(param, self.SpecificHyperparamList.get(param)[self.exp_count])
                return False
            elif self.exp_count == len(self.SpecificHyperparamList.get(list(keys)[0])):
                if self.OnlySpecific:
                    return True
                else:
                    self.HyperparameterInit(self.HyperparameterList)
                    self.result = []
                    return False

        # After trying the given specific params, using greedy search in the AdjustableParamList(HyperParameterList).
        ParamNames = list(self.HyperparameterList.keys())
        cur_param_name = ParamNames[self.parampointer]           # key, string
        cur_param = self.HyperparameterList[cur_param_name]      # list of values
        if self.paramvaluepointer < len(cur_param):
            # set the config
            cur_value = cur_param[self.paramvaluepointer]        # value
            self.opt.set_args(cur_param_name, cur_value)
            self.paramvaluepointer += 1
        else:
            # choose the best param value based on the results.
            assert len(self.result) == len(cur_param)

            if self.opt.args['ClassNum'] == 1:
                best_metric = min(self.result)
            else:
                best_metric = max(self.result)

            loc = self.result.index(best_metric)
            self.result = []
            self.result.append(best_metric)                      # best_metric is obtained by configs: {paraml:[loc], paraml+1:[0]}
                                                                 # so that we don't need to test the choice of paraml+1:[0]
                                                                 # just use the result tested when adjusting paraml.
            cur_param_best_value = cur_param[loc]
            self.opt.set_args(cur_param_name, cur_param_best_value)
            self.parampointer += 1
            self.paramvaluepointer = 1                           # test from paraml+1:[1]

            if self.parampointer < len(ParamNames):
                # set the config
                cur_param_name = ParamNames[self.parampointer]
                cur_param = self.HyperparameterList[cur_param_name]
                cur_value = cur_param[self.paramvaluepointer]
                self.opt.set_args(cur_param_name, cur_value)
                self.paramvaluepointer += 1
                return False
            else:
                return True



    def StoreResults(self, score):
        self.result.append(score)

    def LoadStatusFromFile(self):
        status = self.controllerstatussaver.LoadStatus()
        return status

    def SaveStatusToFile(self):
        self.GetControllerStatus()
        print(self.status)
        self.controllerstatussaver.SaveStatus(self.status)

    def InitControllerStatus(self):
        # running status of the ConfigController.
        self.exp_count = 0
        self.parampointer = 0
        self.paramvaluepointer = 1
        # two pointers indicates the param and its value that next experiment should use.
        self.result = []

    def GetControllerStatus(self):
        self.status = {
            'exp_count': self.exp_count,
            'parampointer': self.parampointer,
            'paramvaluepointer': self.paramvaluepointer,
            'result': self.result,
            'next_opt_args': self.opt.args
        }

    def SetControllerStatus(self, status):
        self.exp_count = status['exp_count']
        self.parampointer = status['parampointer']
        self.paramvaluepointer = status['paramvaluepointer']
        self.result = status['result']
        self.opt.args = status['next_opt_args']
        #print("Config Controller has been loaded. Experiments continue.")

##########################################################################################################

class CkptController(Controller):
    # controller to deal with the check point after each epoch training.
    # A ckpt happens on the end of a training epoch
    # including following information:
    # model: the model after current epoch training
    # optimizer: the optimizer after current epoch training
    # epoch: the epoch number
    # These information should be saved and be loaded and maintained for latter training.
    # scores: the result on valid set
    # testscores: the result on test set
    # These informations should be saved as files and not need to maintain and reload.
    # ckpt_count: The status that should be maintained during training.

    # The status of the ckpt controller is only the ckpt_count and the ckpt_name
    # and other informations are not needed to be maintained in the controller.
    # they should be saved when calling CkptProcessing function, and loaded when calling LoadCkpt function.
    def __init__(self, opt):
        super(CkptController, self).__init__()
        self.opt = opt
        self.ckpt_count = 0
        self.saver = Saver()
        #

    def CkptProcessing(self, model, optimizer, epoch, scores, testscores):
        # Saving the check point to a file.
        ckpt_name = self.opt.args['SaveDir'] + 'model/model_optimizer_epoch' + str(self.ckpt_count)
        ckpt = {
            'model': model,
            'optimizer': optimizer,
            'epoch': epoch
        }
        self.SaveCkpt(ckpt, ckpt_name)
        # Saving the result to a file.
        valid_result_file_name = self.opt.args['SaveDir'] + 'result' + str(self.ckpt_count) + 'valid.json'
        self.saver.SaveContext(valid_result_file_name, scores)
        test_result_file_name = self.opt.args['SaveDir'] + 'result' + str(self.ckpt_count) + 'test.json'
        self.saver.SaveContext(test_result_file_name, testscores)
        print('Results saved.')

        self.ckpt_count += 1

    def SaveCkpt(self, ckpt, ckpt_name):
        t.save(ckpt, ckpt_name)
        print('Model Saved.')

    def LoadCkpt(self):
        last_model_ckpt = self.FindLastCkpt()
        if last_model_ckpt:
            ckpt = t.load(os.path.join(self.opt.args['ModelDir'], last_model_ckpt))
            model = ckpt['model']
            optimizer = ckpt['optimizer']
            epoch = ckpt['epoch']
            return model, optimizer, epoch
        else:
            return None, None, None

    def FindLastCkpt(self):
        dir_files = os.listdir(self.opt.args['ModelDir'])  # list of the ckpt files.
        if dir_files:
            dir_files = sorted(dir_files, key=lambda x: os.path.getctime(os.path.join(self.opt.args['ModelDir'],x)))
            last_model_ckpt = dir_files[-1] # find the latest ckpt file.
            return last_model_ckpt
        else:
            return None

    def GetControllerStatus(self):
        status = {
            'ckpt_count': self.ckpt_count
        }
        return status

    def SetControllerStatus(self, status):
        self.ckpt_count = status['ckpt_count']

##########################################################################################################

class ExperimentProcessController(Controller):
    # Module to control the entire experimental process.
    def __init__(self, ExpOptions, Params):
        # ExpOptions: Options to set the ExperimentProcessController
        # Params: Params for the experiment. i.g. the three ParamLists of greedy search.
        super(ExperimentProcessController, self).__init__()

        self.ExpOptions = ExpOptions
        self.search = self.ExpOptions['Search']
        self.seedperopt = self.ExpOptions['SeedPerOpt']
        self.Kfold = self.ExpOptions['Kfold']
        self.OnlyEval = self.ExpOptions['OnlyEval']
        self.SeperatedTEMode = self.ExpOptions['SeperatedTEMode']
        self.Finetune = self.ExpOptions['Finetune']
        # todo(zqzhang): updated in FraGAT_dissertation_ver
        self.TorchSeedBias = self.ExpOptions['TorchSeedBias']

        # process the params based on different searching methods, determined by the ExpOptions
        if self.search == 'greedy':
            self.BasicParamList, self.AdjustableParamList, self.SpecificParamList = Params

        if self.OnlyEval:
            # todo(zqzhang): updated in TPv7
            self.EvalParams = {'EvalModelPath': self.BasicParamList['EvalModelPath'],
                               'EvalDatasetPath': self.BasicParamList['EvalDatasetPath'],
                               'EvalLogAllPreds': self.BasicParamList['EvalLogAllPreds'],
                               'EvalOptPath': self.BasicParamList['EvalOptPath'],
                               'EvalBothSubsets': self.BasicParamList['EvalBothSubsets']
                          }

        # todo(zqzhang): updated in TPv7
        if self.SeperatedTEMode == 'Eval':
            self.SeperatedEvalParams = {

            }

        self.ConfigControllersList = {
            'greedy': GreedyConfigController
        }

        # os.environ['CUDA_VISIBLE_DEVICES'] = self.BasicParamList['CUDA_VISIBLE_DEVICES']
        self.configcontroller = self.ConfigControllersList[self.search](self.BasicParamList, self.AdjustableParamList, self.SpecificParamList)

        self.controllerstatussaver = ControllerStatusSaver(self.configcontroller.opt.args,'ExperimentProcessController')

        status = self.LoadStatusFromFile()
        if status:
            self.SetControllerStatus(status)
        else:
            self.InitControllerStatus()
        # The status: cur_opt_results, opt_results and i have been set, either initialized or loaded from files.

    def ExperimentStart(self):
        # Set the Config Controllers
        if self.OnlyEval:
            # todo(zqzhang): updated in TPv7
            tmp_opt = self.configcontroller.GetOpts()
            tmp_opt.args.update(self.EvalParams)
            # print(opt.args)
            self.onlyevaler = OnlyEvaler(tmp_opt)
            self.onlyevaler.OnlyEval()
            return 0

        # todo(zqzhang): updated in TPv7
        if self.SeperatedTEMode == 'Eval':
            tmp_opt = self.configcontroller.GetOpts()
            print(tmp_opt.args)
            # tmp_opt.args.update(self.SeperatedEvalParams)
            # opt.args.update()
            self.SeperatedEvaler = SeperatedEvaler(tmp_opt)
            self.SeperatedEvaler.Eval()
            return 0

        # todo(zqzhang): updated in TPv7
        if self.SeperatedTEMode == 'Train':
            no_eval_flag = True
        else:
            no_eval_flag = False

        end_flag = False

        while not end_flag:
            opt = self.configcontroller.GetOpts()

            while self.i < self.seedperopt:
                self.CheckDirectories(opt, self.i)
                opt.set_args('TorchSeed', self.i + self.TorchSeedBias)

                print("The parameters of the current exp are: ")
                print(opt.args)

                if self.Finetune:
                    trainer = FinetuneTrainer(opt)
                else:
                    trainer = Trainer(opt,self.Kfold, no_eval_flag)

                if not self.Kfold:
                    ckpt, value = trainer.TrainOneOpt()
                else:
                    ckpts, values = trainer.TrainOneOpt_KFold()
                    value = np.mean(values)

                print(f"cur_opt_cur_seed_value: {value}")

                self.cur_opt_results.append(value)
                self.i += 1
                self.SaveStatusToFile()
                self.configcontroller.SaveStatusToFile()

            cur_opt_value = np.mean(self.cur_opt_results)     # the average result value of the current opt on self.seedperopt times running.
            self.opt_results.append(cur_opt_value)
            self.cur_opt_results = []                         # clear the buffer of current opt results.

            self.configcontroller.StoreResults(cur_opt_value)
            self.configcontroller.exp_count += 1
            end_flag = self.configcontroller.AdjustParams()
            self.i = 0

            if self.ExpOptions['SeedSearch']:
                end_flag = True

        print("Experiment Finished")
        print("The best averaged value of all opts is: ")
        if opt.args['ClassNum'] == 1:
            best_opt_result = min(self.opt_results)
            print(best_opt_result)
        else:
            best_opt_result = max(self.opt_results)
            print(best_opt_result)
        print("And the corresponding exp num is: ")
        print(self.opt_results.index(best_opt_result))

    def CheckDirectories(self, opt, i):
        opt.set_args('SaveDir', opt.args['ExpDir'] + str(i) + '/')
        if not os.path.exists(opt.args['SaveDir']):
            os.mkdir(opt.args['SaveDir'])

        opt.set_args('ModelDir', opt.args['SaveDir'] + 'model/')
        if not os.path.exists(opt.args['ModelDir']):
            os.mkdir(opt.args['ModelDir'])

    def LoadStatusFromFile(self):
        status = self.controllerstatussaver.LoadStatus()
        return status

    def SaveStatusToFile(self):
        self.GetControllerStatus()
        print(self.status)
        self.controllerstatussaver.SaveStatus(self.status)

    def InitControllerStatus(self):
        self.cur_opt_results = []
        self.opt_results = []
        self.i = 0

    def GetControllerStatus(self):
        self.status = {
            'cur_opt_results': self.cur_opt_results,
            'opt_results': self.opt_results,
            'cur_i': self.i,
        }

    def SetControllerStatus(self, status):
        self.cur_opt_results = status['cur_opt_results']
        self.opt_results = status['opt_results']
        self.i = status['cur_i']
        assert self.i == len(self.cur_opt_results)

class OnlyEvaler(Controller):
    # todo(zqzhang): updated in TPv7.8
    def __init__(self, tmp_opt, evalloaders = None):
        super(OnlyEvaler, self).__init__()
        self.tmp_opt = tmp_opt
        # print(self.opt.args)
        self.EvalModelPath = self.tmp_opt.args['EvalModelPath']
        self.EvalDatasetPath = self.tmp_opt.args['EvalDatasetPath']
        self.LogAllPreds = self.tmp_opt.args['EvalLogAllPreds']
        self.EvalOptPath = self.tmp_opt.args['EvalOptPath']
        self.EvalBothSubsets = self.tmp_opt.args['EvalBothSubsets']

        self.Saver = Saver()

        self.LoadEvalOpt()

        if not evalloaders:
            self.BuildEvalDataset()
        else:
            self.evalloaders = evalloaders

        self.BuildEvalModel()
        self.evaluator = self.BuildEvaluator()

    # todo(zqzhang): updated in TPv7
    def LoadEvalOpt(self):
        args = self.Saver.LoadContext(self.EvalOptPath)
        self.tmp_opt.args.update(args)
        self.opt = self.tmp_opt

    # todo(zqzhang): updated in TPv7
    def BuildEvalDataset(self):
        if self.EvalDatasetPath:
            file_loader = JsonFileLoader(self.EvalDatasetPath)
            eval_dataset = file_loader.load()
            Evalset = MolDataset(eval_dataset)

        else:
            moldatasetcreator = MolDatasetCreator(self.opt)
            sets, _ = moldatasetcreator.CreateDatasets()
            if len(self.opt.args['SplitRate']) == 2:
                (Trainset, Validset, Testset) = sets
                if self.EvalBothSubsets:
                    self.Evalset1 = Validset
                    self.Evalset2 = Testset
            elif len(self.opt.args['SplitRate']) == 1:
                (Trainset, Validset) = sets
                self.Evalset = Validset
            else:
                raise RuntimeError("No subset to be evaluated!")

        self.batchsize = self.opt.args['BatchSize']
        if not self.opt.args['PyG']:
            if self.opt.args['Model'] == 'Graphormer':
                if self.EvalBothSubsets:
                    self.evalloaders = [t.utils.data.DataLoader(self.Evalset1, batch_size = self.batchsize, shuffle = False,
                                                           num_workers = 8,
                                                           drop_last = False, worker_init_fn = np.random.seed(8),
                                                           pin_memory = False,
                                                           collate_fn = partial(collator,
                                                                                max_node = self.opt.args['max_node'],
                                                                                multi_hop_max_dist = self.opt.args[
                                                                                    'multi_hop_max_dist'],
                                                                                spatial_pos_max = self.opt.args[
                                                                                    'spatial_pos_max'])
                                                           ),
                                        t.utils.data.DataLoader(self.Evalset2, batch_size = self.batchsize, shuffle = False,
                                                          num_workers = 8,
                                                          drop_last = False, worker_init_fn = np.random.seed(8),
                                                          pin_memory = False,
                                                          collate_fn = partial(collator,
                                                                               max_node = self.opt.args['max_node'],
                                                                               multi_hop_max_dist = self.opt.args[
                                                                                   'multi_hop_max_dist'],
                                                                               spatial_pos_max = self.opt.args[
                                                                                   'spatial_pos_max'])
                                                          )
                    ]
                else:
                    self.evalloader = t.utils.data.DataLoader(self.Evalset, batch_size = self.batchsize, shuffle = False,
                                                          num_workers = 8,
                                                          drop_last = False, worker_init_fn = np.random.seed(8),
                                                          pin_memory = False,
                                                          collate_fn = partial(collator,
                                                                               max_node = self.opt.args['max_node'],
                                                                               multi_hop_max_dist = self.opt.args[
                                                                                   'multi_hop_max_dist'],
                                                                               spatial_pos_max = self.opt.args[
                                                                                   'spatial_pos_max'])
                                                          )
            else:
                if self.EvalBothSubsets:
                    self.evalloaders = [
                        t.utils.data.DataLoader(self.Evalset1, batch_size = self.batchsize, shuffle = False,
                                                num_workers = 8,
                                                drop_last = False, worker_init_fn = np.random.seed(8),
                                                pin_memory = False),
                        t.utils.data.DataLoader(self.Evalset2, batch_size = self.batchsize, shuffle = False,
                                                num_workers = 8, \
                                                drop_last = False, worker_init_fn = np.random.seed(8),
                                                pin_memory = False)
                        ]
                else:
                    self.evalloader = t.utils.data.DataLoader(self.Evalset, batch_size = self.batchsize,
                                                              shuffle = False, num_workers = 8,
                                                              drop_last = False, worker_init_fn = np.random.seed(8),
                                                              pin_memory = False)
                ####
        else:
            import torch_geometric as tg
            if self.EvalBothSubsets:
                self.evalloaders = [tg.loader.DataLoader(self.Evalset1, batch_size = 8, shuffle = False, num_workers = 2, \
                                                    drop_last = False, worker_init_fn = np.random.seed(8),
                                                    pin_memory = True),
                                    tg.loader.DataLoader(self.Evalset2, batch_size = 8, shuffle = False, num_workers = 2, \
                                                         drop_last = False, worker_init_fn = np.random.seed(8),
                                                         pin_memory = True)]
            else:
                self.evalloader = tg.loader.DataLoader(self.Evalset, batch_size = 8, shuffle = False, num_workers = 2, \
                                                         drop_last = False, worker_init_fn = np.random.seed(8),
                                                         pin_memory = True)

            # print(f"Trainloader: {self.trainloader}")
            # print(f"Validloader: {self.validloader}")
            # print(f"Testloader: {self.testloader}")

    def BuildEvalModel(self):
        ckpt = None
        while not ckpt:
            try:
                ckpt = t.load(self.EvalModelPath)
            except:
                time.sleep(1)
        self.net = ckpt['model']

    def BuildEvaluator(self):
        if self.opt.args['Model'] == 'FraGAT':
            evaluator = FraGATEvaluator(self.opt)

        # todo(zqzhang): updated in TPv7
        elif self.opt.args['PyG']:
            evaluator = PyGEvaluator(self.opt)
        else:
            evaluator = GeneralEvaluator(self.opt)
        return evaluator

    # todo(zqzhang): updated in TPv7
    def OnlyEval(self):
        if self.opt.args['ClassNum'] == 1:
            if self.EvalBothSubsets:
                print('Running on Validset')
                evalloader = self.evalloaders[0]
                validresult = self.evaluator.eval(evalloader, self.net, [MAE(), RMSE()])
                print('Running on Testset')
                evalloader = self.evalloaders[1]
                testresult = self.evaluator.eval(evalloader, self.net, [MAE(), RMSE()])
            else:
                print('Running on Evalset')
                result = self.evaluator.eval(self.evalloader, self.net, [MAE(), RMSE()])

        else:
            if self.EvalBothSubsets:
                print('Running on Validset')
                evalloader = self.evalloaders[0]
                validresult = self.evaluator.eval(evalloader, self.net, [AUC(), ACC()])
                print('Running on Testset')
                evalloader = self.evalloaders[1]
                testresult = self.evaluator.eval(evalloader, self.net, [AUC(), ACC()])
            else:
                print("Running on Evalset")
                result = self.evaluator.eval(self.evalloader, self.net, [AUC(), ACC()])

        if self.LogAllPreds:
            print("All samples in Evalset and their predictions:")

            Answers = []
            for i in range(self.opt.args['TaskNum']):
                Answers.append([])

            for i in range(self.opt.args['TaskNum']):
                for j in range(len(self.Evalset.dataset)):
                    item = self.Evalset.dataset[j]
                    SMILES = item['SMILES']
                    Label = item['Value'][i]
                    # print(self.evaluator.AllLabel)
                    Label_e = self.evaluator.AllLabel[i][j]

                    # print(Label)
                    # print(Label_e)

                    assert Label == str(Label_e)
                    pred = self.evaluator.AllPred[i][j]
                    Answers[i].append({'SMILES': SMILES, 'Value': Label, 'Pred': pred})

            print(Answers)

            Addr = self.opt.args['SaveDir']+'AllAnswers.json'
            with open(Addr,'w') as f:
                json.dump(Answers, f)

        # todo(zqzhang): updated in TPv7
        if self.EvalBothSubsets:
            return (validresult, testresult)
        else:
            return result


# todo(zqzhang): updated in TPv7
class SeperatedEvaler(Controller):
    def __init__(self, tmp_opt):
        super(SeperatedEvaler, self).__init__()
        self.tmp_opt = tmp_opt

        # Dir control
        self.Trial_Path = self.tmp_opt.args['RootPath'] + self.tmp_opt.args['ExpName'] +'/'

        self.saver = Saver()
        if self.tmp_opt.args['ClassNum'] == 1:
            self.BestValidRMSE = 1e9
        else:
            self.BestValidAUC = 0.0

    def Eval(self):
        print(f"TrialPath: {self.Trial_Path}")
        exp_end_flag = 0
        last_seed_dir = None
        while not exp_end_flag:

            cur_exp_name = self.FindCurExp()
            CurExpDir = self.Trial_Path + cur_exp_name + '/'
            cur_seed_name = self.LastSeedName(CurExpDir)
            while cur_seed_name == ' ':
                cur_seed_name = self.LastSeedName(CurExpDir)
            CurSeedDir = CurExpDir + cur_seed_name + '/'
            print(f'Current Seed Dir: {CurSeedDir}')

            if CurSeedDir == last_seed_dir:
                break

            self.CurSeedDir = CurSeedDir


            cur_opt_file = CurSeedDir + 'config.json'
            cur_args = None
            while not cur_args:
                try:
                    cur_args = self.saver.LoadContext(cur_opt_file)
                except:
                    time.sleep(1)
            # while not os.path.exists(cur_opt_file):
            #     a = 1
            # cur_args = self.saver.LoadContext(cur_opt_file)
            self.tmp_opt.args.update(cur_args)
            print(f"cur_opt: {cur_args}")

            self.earlystopcontroller = EarlyStopController(self.tmp_opt)
            self.EvalOneOpt(cur_opt_file)

            if self.tmp_opt.args['ClassNum'] == 1:
                print(f"The best valid result of the current opt is: {self.BestValidRMSE} ")
                print(f"And its test result is: {self.TestRMSEofBestValid}")
                self.SaveCurOptBestResult(self.BestValidRMSE, self.TestRMSEofBestValid)
                self.BestValidRMSE = 1e9
            else:
                print(f"The best valid result of the current opt is: {self.BestValidAUC} ")
                print(f"And its test result is: {self.TestAUCofBestValid}")
                self.SaveCurOptBestResult(self.BestValidAUC, self.TestAUCofBestValid)
                self.BestValidAUC = 0

            last_seed_dir = CurSeedDir
            time.sleep(10)

    def SaveCurOptBestResult(self, best_valid, test_of_best_valid):
        savedir = self.CurSeedDir + 'cur_opt_best_results.json'
        best_result = {
            'best_valid': best_valid,
            'test_of_best_valid': test_of_best_valid
        }
        self.saver.SaveContext(savedir, best_result)

    def EvalOneOpt(self, cur_opt_file):
        max_epoch_num = self.tmp_opt.args['MaxEpoch']

        stop_flag = 0

        self.BuildEvalDataset()

        for i in range(max_epoch_num):

            if stop_flag:
                MaxResult = self.earlystopcontroller.MaxResult
                BestModel = self.earlystopcontroller.MaxResultModelIdx
                print("Early Stop")
                print("The Best Result is: ")
                print(MaxResult)
                print("and its corresponding model ckpt is: ")
                print(BestModel)
                self.WaitForTrainerStop()
                self.RemoveOtherCkpts(BestModel)
                break

            cur_model_dir = self.CurSeedDir + 'model/'
            cur_model_name = 'model' + str(i)
            cur_model = cur_model_dir + cur_model_name
            while not os.path.exists(cur_model):
                a = 1

            print(f'Evaluating on model ckpt: {cur_model_name}')
            validresult, testresult = self.EvalOneModel(cur_model, cur_opt_file)

            if self.tmp_opt.args['ClassNum'] == 1:
                valid_result_rmse = validresult['RMSE']
                test_result_rmse = testresult['RMSE']
                if valid_result_rmse < self.BestValidRMSE:
                    self.BestValidRMSE = valid_result_rmse
                    self.TestRMSEofBestValid = test_result_rmse

                print('Best Valid: ')
                print(self.BestValidRMSE)
                print('Best Test: ')
                print(self.TestRMSEofBestValid)
                self.SaveResultCkpt(i, valid_result_rmse, test_result_rmse)
                stop_flag = self.earlystopcontroller.ShouldStop(validresult, i, testresult)

            else:
                valid_result_auc = validresult['AUC']
                test_result_auc = testresult['AUC']
                if valid_result_auc > self.BestValidAUC:
                    self.BestValidAUC = valid_result_auc
                    self.TestAUCofBestValid = test_result_auc

                print('Best Valid: ')
                print(self.BestValidAUC)
                print('Best Test: ')
                print(self.TestAUCofBestValid)
                self.SaveResultCkpt(i, valid_result_auc, test_result_auc)
                stop_flag = self.earlystopcontroller.ShouldStop(validresult, i, testresult)

            stop_flag_file = self.CurSeedDir + 'stop_flag.json'
            self.saver.SaveContext(stop_flag_file, stop_flag)

        MaxResult = self.earlystopcontroller.MaxResult
        BestModel = self.earlystopcontroller.MaxResultModelIdx
        print("Approaching Max Epoch.")
        print("The Best Result is: ")
        print(MaxResult)
        print("and its corresponding model ckpt is: ")
        print(BestModel)
        self.RemoveOtherCkpts(BestModel)

    def BuildEvalDataset(self):
        moldatasetcreator = MolDatasetCreator(self.tmp_opt)
        sets, _ = moldatasetcreator.CreateDatasets()
        if len(self.tmp_opt.args['SplitRate']) == 2:
            (Trainset, Validset, Testset) = sets
            self.Evalset1 = Validset
            self.Evalset2 = Testset
        elif len(self.tmp_opt.args['SplitRate']) == 1:
            (Trainset, Validset) = sets
            self.Evalset = Validset
        else:
            raise RuntimeError("No subset to be evaluated!")

        self.batchsize = self.tmp_opt.args['BatchSize']
        if not self.tmp_opt.args['PyG']:
            if self.tmp_opt.args['Model'] == 'Graphormer':
                from Models.Graphormer.collator import collator
                self.evalloaders = [
                        t.utils.data.DataLoader(self.Evalset1, batch_size = self.batchsize, shuffle = False,
                                                num_workers = 8,
                                                drop_last = False, worker_init_fn = np.random.seed(8),
                                                pin_memory = False,
                                                collate_fn = partial(collator,
                                                                     max_node = self.tmp_opt.args['max_node'],
                                                                     multi_hop_max_dist = self.tmp_opt.args[
                                                                         'multi_hop_max_dist'],
                                                                     spatial_pos_max = self.tmp_opt.args[
                                                                         'spatial_pos_max'])
                                                ),
                        t.utils.data.DataLoader(self.Evalset2, batch_size = self.batchsize, shuffle = False,
                                                num_workers = 8,
                                                drop_last = False, worker_init_fn = np.random.seed(8),
                                                pin_memory = False,
                                                collate_fn = partial(collator,
                                                                     max_node = self.tmp_opt.args['max_node'],
                                                                     multi_hop_max_dist = self.tmp_opt.args[
                                                                         'multi_hop_max_dist'],
                                                                     spatial_pos_max = self.tmp_opt.args[
                                                                         'spatial_pos_max'])
                                                )
                        ]

            else:
                self.evalloaders = [
                        t.utils.data.DataLoader(self.Evalset1, batch_size = self.batchsize, shuffle = False,
                                                num_workers = 8,
                                                drop_last = False, worker_init_fn = np.random.seed(8),
                                                pin_memory = False),
                        t.utils.data.DataLoader(self.Evalset2, batch_size = self.batchsize, shuffle = False,
                                                num_workers = 8, \
                                                drop_last = False, worker_init_fn = np.random.seed(8),
                                                pin_memory = False)
                    ]
                ####
        else:
            # TODO(ailin): update for FragTransformer
            if self.tmp_opt.args['Model'] == 'FragTransformer':
                from FragComponents.FragTransformer.collator import collator
                self.evalloaders = [
                        t.utils.data.DataLoader(self.Evalset1, batch_size = self.batchsize, shuffle = False,
                                                num_workers = 8,
                                                drop_last = False, worker_init_fn = np.random.seed(8),
                                                pin_memory = False,
                                                collate_fn = partial(collator,
                                                                     max_node = self.tmp_opt.args['max_node'],
                                                                     multi_hop_max_dist = self.tmp_opt.args[
                                                                         'multi_hop_max_dist'],
                                                                     spatial_pos_max = self.tmp_opt.args[
                                                                         'spatial_pos_max'])
                                                ),
                        t.utils.data.DataLoader(self.Evalset2, batch_size = self.batchsize, shuffle = False,
                                                num_workers = 8,
                                                drop_last = False, worker_init_fn = np.random.seed(8),
                                                pin_memory = False,
                                                collate_fn = partial(collator,
                                                                     max_node = self.tmp_opt.args['max_node'],
                                                                     multi_hop_max_dist = self.tmp_opt.args[
                                                                         'multi_hop_max_dist'],
                                                                     spatial_pos_max = self.tmp_opt.args[
                                                                         'spatial_pos_max'])
                                                )
                        ]
            else:
                import torch_geometric as tg
                self.evalloaders = [
                        tg.loader.DataLoader(self.Evalset1, batch_size = 8, shuffle = False, num_workers = 2, \
                                             drop_last = False, worker_init_fn = np.random.seed(8),
                                             pin_memory = True),
                        tg.loader.DataLoader(self.Evalset2, batch_size = 8, shuffle = False, num_workers = 2, \
                                             drop_last = False, worker_init_fn = np.random.seed(8),
                                             pin_memory = True)]

            # print(f"Trainloader: {self.trainloader}")
            # print(f"Validloader: {self.validloader}")
            # print(f"Testloader: {self.testloader}")

    def EvalOneModel(self, last_model, cur_opt_file):
        cur_evaler_params = {
            'EvalModelPath': last_model,
            'EvalDatasetPath': None,
            'EvalBothSubsets': True,
            'EvalLogAllPreds': False,
            'EvalOptPath': cur_opt_file,
        }
        self.tmp_opt.args.update(cur_evaler_params)
        self.onlyevaler = OnlyEvaler(self.tmp_opt, self.evalloaders)
        (validresult, testresult) = self.onlyevaler.OnlyEval()
        return validresult, testresult

    def SaveResultCkpt(self, ckpt_idx, valid_result, test_result = None):
        results = {'cur_opt_valid': valid_result,
                   'cur_opt_test': test_result}
        addr = self.CurSeedDir + 'results/'
        self.CheckDirectory(addr)

        result_ckpt_name = addr + 'result' + str(ckpt_idx)
        self.saver.SaveContext(result_ckpt_name, results)
        print('Result saved!')

    def CheckDirectory(self, addr):
        if not os.path.exists(addr):
            os.mkdir(addr)

    def FindCurExp(self):
        dir_names = os.listdir(self.Trial_Path)
        cnt = []
        for dir in dir_names:
            if re.match('exp',dir):
                num = re.split('exp', dir)[-1]
                cnt.append(int(num))
        cur_exp = 'exp' + str(max(cnt))
        return cur_exp


    def LastSeedName(self, Addr):
        dir_files = os.listdir(Addr)
        if dir_files:
            int_dir_files = []
            for ele in dir_files:
                int_dir_files.append(int(ele))
            last_file = str(max(int_dir_files))
        else:
            last_file = ' '
        # print(f"last_file: {last_file}")
        return last_file

    def LastFileName(self, Addr):
        dir_files = os.listdir(Addr)
        # print(f"dir_files: {dir_files}")
        # os.listdir returns the file names in Addr, only the names, without the Addr path.
        if dir_files:
            dir_files = sorted(dir_files, key=lambda x: os.path.getctime(os.path.join(Addr, x)))
            last_file = dir_files[-1]
        else:
            last_file = ' '
        # print(f"last_file: {last_file}")
        return last_file

    def RemoveOtherCkpts(self, bestmodel):
        if bestmodel == None:
            print(f"Ckpts will be deleted by Seperated Evaler.")
            return 0

        print(f"Deleting other ckpt models.")
        model_dir = self.CurSeedDir + 'model/'
        filenames = os.listdir(model_dir)
        for file in filenames:
            if file != ('model' + str(bestmodel)):
                os.remove(model_dir + file)

        print(f"Deleting other result files.")
        result_dir = self.CurSeedDir + 'results/'
        filenames = os.listdir(result_dir)
        for file in filenames:
            if file != ('result' + str(bestmodel)):
                os.remove(result_dir + file)

        print(f"Deleting other TrainerStatus files.")
        status_dir = self.CurSeedDir + 'TrainerStatus/'
        filenames = os.listdir(status_dir)
        filename = self.LastFileName(status_dir)
        for file in filenames:
            if file != filename:
                os.remove(status_dir + file)

    def WaitForTrainerStop(self):
        print(f"Waiting for trainer stopping.")
        trainer_stop_flag_file = self.CurSeedDir + 'trainer_stop_flag.json'
        while not os.path.exists(trainer_stop_flag_file):
            a = 1
        trainer_stop_flag = self.saver.LoadContext(trainer_stop_flag_file)
        assert trainer_stop_flag == 1
        return


##########################################################################################################

class Trainer(object):
    # todo(zqzhang): updated in TPv7
    def __init__(self, opt, KFold=False, no_eval_flag=False):
        super(Trainer, self).__init__()
        self.opt = opt
        self.KFold = KFold
        # self.train_fn = self.TrainOneEpoch
        self.device = t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')
        # self.device = t.device('cpu')

        # todo(zqzhang): updated in TPv7
        self.no_eval_flag = no_eval_flag

        t.manual_seed(self.opt.args['TorchSeed'])
        #statussaver = ControllerStatusSaver(self.opt.args, 'Trainer')
        # todo(ailin)
        np.random.seed(self.opt.args['TorchSeed'])
        random.seed(self.opt.args['TorchSeed'])
        os.environ['PYTHONHASHSEED'] = str(self.opt.args['TorchSeed'])
        # os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        t.cuda.manual_seed(self.opt.args['TorchSeed'])
        t.cuda.manual_seed_all(self.opt.args['TorchSeed'])
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        # t.use_deterministic_algorithms(True)

        self.BuildDataset()

        self.net = self.BuildModel()
        self.BuildIniter()
        if self.initer:
            self.WeightInit()

        self.BuildOptimizer()

        self.StartEpoch = 0

        self.lr_sch = self.BuildScheduler()
        self.BuildCriterion()
        self.evaluator = self.BuildEvaluator()
        self.earlystopcontroller = EarlyStopController(self.opt)


        self.saver = Saver()
        self.controllerstatussaver = ControllerStatusSaver(self.opt.args, 'Trainer', restart=True)

    def reinit(self):
        # self.device = t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')
        self.device = t.device('cpu')
        self.net = self.BuildModel()
        self.BuildIniter()
        if self.initer:
            self.WeightInit()

        self.BuildOptimizer()
        self.StartEpoch = 0

        self.lr_sch = self.BuildScheduler()
        self.BuildCriterion()
        self.evaluator = self.BuildEvaluator()
        self.earlystopcontroller = EarlyStopController(self.opt)


        self.saver = Saver()
        self.controllerstatussaver = ControllerStatusSaver(self.opt.args, 'Trainer', restart = True)

    ########################################################
    # todo(zqzhang): updated in TPv7
    def BuildModel(self):
        if self.opt.args['Model'] == 'FraGAT':
            net = MolPredFraGAT(
                    atom_feature_size = self.opt.args['AtomFeatureSize'],
                    bond_feature_size = self.opt.args['BondFeatureSize'],
                    FP_size = self.opt.args['FPSize'],
                    atom_layers = self.opt.args['AtomLayers'],
                    mol_layers = self.opt.args['MolLayers'],
                    DNN_layers = self.opt.args['DNNLayers'],
                    output_size = self.opt.args["OutputSize"],
                    droprate = self.opt.args['DropRate'],
                    opt = self.opt
            ).to(self.device)

        elif self.opt.args['Model'] == 'PyGFraGAT':
            net = PyGFraGAT(self.opt).to(self.device)

        # elif self.opt.args['Model'] == 'FraContra':
        #     net = FraContra(self.opt).to(self.device)

        elif self.opt.args['Model'] == 'CMPNN':
            net = CMPNNModel(
                    self.opt.args['dataset_type']=='classification',
                    self.opt.args['dataset_type']=='multiclass',
                    opt = self.opt).to(self.device)

        elif self.opt.args['Model'] == 'PyGGCN':
            net = PyGGCN(self.opt).to(self.device)

        elif self.opt.args['Model'] == 'PyGGIN':
            net = PyGGIN(self.opt).to(self.device)

        elif self.opt.args['Model'] == 'PyGSGC':
            net = PyGSGC(self.opt).to(self.device)

        elif self.opt.args['Model'] == 'Graphormer':
            net = Graphormer(
                    num_encoder_layers = self.opt.args['num_encoder_layers'],
                    num_attention_heads = self.opt.args['num_attention_heads'],
                    embedding_dim = self.opt.args['embedding_dim'],
                    dropout_rate = self.opt.args['dropout_rate'],
                    intput_dropout_rate = self.opt.args['intput_dropout_rate'],
                    ffn_dim = self.opt.args['ffn_dim'],
                    edge_type = self.opt.args['edge_type'],
                    multi_hop_max_dist = self.opt.args['multi_hop_max_dist'],
                    attention_dropout_rate = self.opt.args['attention_dropout_rate'],
                    flag = self.opt.args['flag'],
                    opt = self.opt
            # ).to(t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu'))
            ).to(self.device)

        elif self.opt.args['Model'] == 'MLP':
            if self.opt.args['Toyset']:
                net = DNN(
                        self.opt.args['InputFeatureSize'],
                        self.opt.args['DNNLayers'],
                        self.opt.args['OutputSize'],
                        self.opt
                ).to(self.device)

        # todo(zqzhang): updated in TPv8
        elif self.opt.args['Model'] == 'PyGATFP':
            net = PyGATFP(self.opt).to(self.device)

        elif self.opt.args['Model'] == 'PyGGraphSAGE':
            net = PyGGraphSAGE(self.opt).to(self.device)

        elif self.opt.args['Model'] == 'LSTM':
            net = MolPredLSTM(self.opt).to(self.device)

        elif self.opt.args['Model'] == 'GRU':
            net = MolPredGRU(self.opt).to(self.device)

        else:
            raise NotImplementedError

        return net

    def BuildIniter(self):
        init_type = self.opt.args['WeightIniter']
        if init_type == 'Norm':
            self.initer = NormalInitializer(self.opt)

        elif init_type == 'XavierNorm':
            self.initer = XavierNormalInitializer()

        else:
            self.initer = None

    def BuildScheduler(self):
        if self.opt.args['Scheduler'] == 'EmptyLRScheduler':
            lr_sch = EmptyLRSchedular(self.optimizer, lr=10 ** -self.opt.args['lr'])

        elif self.opt.args['Scheduler'] == 'PolynomialDecayLR':
            # tot_updates = self.TrainsetLength * self.opt.args['MaxEpoch'] / self.opt.args['BatchSize']
            # warmup_updates = tot_updates / self.opt.args['WarmupRate']
            warmup_updates = self.opt.args['WarmupEpoch']
            tot_updates = self.opt.args['LRMaxEpoch']
            # warmup_updates = self.opt.args['WarmupUpdates']
            # tot_updates = self.opt.args['TotUpdeates']
            lr = 10 ** -self.opt.args['lr']
            end_lr = self.opt.args['EndLR']
            power = self.opt.args['Power']
            lr_sch = PolynomialDecayLR(self.optimizer, warmup_updates, tot_updates, lr, end_lr, power)

        elif self.opt.args['Scheduler'] == 'StepLR':
            step_size = self.opt.args['LRStep']
            gamma = self.opt.args['LRGamma']
            lr_sch = t.optim.lr_scheduler.StepLR(self.optimizer, step_size, gamma)

        return lr_sch

    def BuildEvaluator(self):
        if self.opt.args['Model'] == 'FraGAT':
            evaluator = FraGATEvaluator(self.opt)
        elif self.opt.args['PyG']:
            evaluator = PyGEvaluator(self.opt)
        else:
            evaluator = GeneralEvaluator(self.opt)
        return evaluator

    def BuildDataset(self):

        if not self.KFold:
            # todo(zqzhang): updated in TPv7
            if self.opt.args['Toyset']:
                toysetcreator = ToysetCreator(self.opt)
                sets, self.weights = toysetcreator.CreateDatasets()
            else:
                moldatasetcreator = MolDatasetCreator(self.opt)
                sets, self.weights = moldatasetcreator.CreateDatasets()

            # print(f"weights: {self.weights}")
            if len(self.opt.args['SplitRate']) == 2:
                (Trainset, Validset, Testset) = sets
            elif len(self.opt.args['SplitRate']) == 1:
                (Trainset, Validset) = sets
                # todo(zqzhang): updated in TPv7
                Testset = []
            else:
                (Trainset) = sets
                Validset = []
                Testset = []

            self.batchsize = self.opt.args['BatchSize']
            if not self.opt.args['PyG']:
                self.TrainsetLength = len(Trainset)
            # print(len(Trainset))
                assert len(Trainset) >= self.batchsize

                if self.opt.args['Model'] == 'Graphormer':
                    self.trainloader = t.utils.data.DataLoader(Trainset, batch_size=self.batchsize, shuffle=True,
                                                               num_workers=8,
                                                               drop_last=True, worker_init_fn=np.random.seed(8),
                                                               pin_memory=True,
                                                               collate_fn=partial(collator,
                                                                                  max_node=self.opt.args['max_node'],
                                                                                  multi_hop_max_dist=self.opt.args[
                                                                                      'multi_hop_max_dist'],
                                                                                  spatial_pos_max=self.opt.args[
                                                                                      'spatial_pos_max'])
                                                               )
                    self.validloader = t.utils.data.DataLoader(Validset, batch_size=self.batchsize, shuffle=False,
                                                               num_workers=8,
                                                               drop_last=False, worker_init_fn=np.random.seed(8),
                                                               pin_memory=False,
                                                               collate_fn=partial(collator,
                                                                                  max_node=self.opt.args['max_node'],
                                                                                  multi_hop_max_dist=self.opt.args[
                                                                                      'multi_hop_max_dist'],
                                                                                  spatial_pos_max=self.opt.args[
                                                                                      'spatial_pos_max'])
                                                               )
                    self.testloader = t.utils.data.DataLoader(Testset, batch_size=self.batchsize, shuffle=False,
                                                              num_workers=8,
                                                              drop_last=False, worker_init_fn=np.random.seed(8),
                                                              pin_memory=False,
                                                              collate_fn=partial(collator,
                                                                                 max_node=self.opt.args['max_node'],
                                                                                 multi_hop_max_dist=self.opt.args[
                                                                                     'multi_hop_max_dist'],
                                                                                 spatial_pos_max=self.opt.args[
                                                                                     'spatial_pos_max'])
                                                              )
                else:
                ###
                    self.trainloader = t.utils.data.DataLoader(Trainset, batch_size = self.batchsize, shuffle = True, num_workers = 0, \
                                              drop_last = True, worker_init_fn = np.random.seed(8), pin_memory = True)
                    self.validloader = t.utils.data.DataLoader(Validset, batch_size = 1, shuffle = False, num_workers = 0, \
                                              drop_last = False, worker_init_fn = np.random.seed(8), pin_memory = True)
                    if len(self.opt.args['SplitRate']) == 2:
                        self.testloader = t.utils.data.DataLoader(Testset, batch_size = 1, shuffle = False, num_workers = 0, \
                                             drop_last = False, worker_init_fn = np.random.seed(8), pin_memory = True)
                    else:
                        self.testloader = None
                    ####
            else:
                import torch_geometric as tg
                self.TrainsetLength = len(Trainset)
                # print(Trainset)
                # print(len(Trainset))
                self.trainloader = tg.loader.DataLoader(Trainset, batch_size = self.batchsize, shuffle = True, num_workers = 8,\
                                                      drop_last = True, worker_init_fn = np.random.seed(8), pin_memory = True)
                self.validloader = tg.loader.DataLoader(Validset, batch_size = 8, shuffle = False, num_workers = 2, \
                                                      drop_last = False, worker_init_fn = np.random.seed(8), pin_memory = True)
                if len(self.opt.args['SplitRate']) == 2:
                    self.testloader = tg.loader.DataLoader(Testset, batch_size = 8, shuffle = False, num_workers = 2, \
                                                         drop_last = False, worker_init_fn = np.random.seed(8), pin_memory = True)
                else:
                    self.testloader = None
                # print(f"Trainloader: {self.trainloader}")
                # print(f"Validloader: {self.validloader}")
                # print(f"Testloader: {self.testloader}")

        # v3 updated
        if self.KFold:
            split_rate = 1 / self.KFold

            self.splitted_sets = []
            for i in range(self.KFold-1):
                self.opt.set_args('SplitRate',[1-split_rate])
                print(self.opt.args['SplitRate'])
                moldatasetcreator = MolDatasetCreator(self.opt)
                if i == 0:
                    sets, self.weights = moldatasetcreator.CreateDatasets()
                else:
                    #print(RemainedSet.dataset)
                    sets, _ = moldatasetcreator.CreateDatasets(RemainedSet.dataset)
                (RemainedSet, ValidSet) = sets
                self.splitted_sets.append(ValidSet)
                if i == self.KFold -2:
                    self.splitted_sets.append(RemainedSet)
                split_rate = 1 / (1/split_rate - 1)

            print(f'For {self.KFold}-fold training, the dataset is splitted into:')
            self.TrainsetLength = 0
            for item in self.splitted_sets:
                print(len(item))
                self.TrainsetLength += len(item)

            self.batchsize = self.opt.args['BatchSize']

            self.TrainsetLength = self.TrainsetLength - len(self.splitted_sets[0])
            # for the first fold, the first subset is used as valid set. So the length of the first subset should be deleted.

        #

    def BuildDataset_KFold(self, FoldNum):
        # v3 updated
        # This function is used to extract one subset from the splitted sets as ValidSet, and merge other subsets as TrainSet
        Validset = self.splitted_sets[FoldNum]
        remained_sets = self.splitted_sets.copy()
        remained_sets.remove(Validset)

        # merge
        merged_set = []
        total_num = 0
        for set in remained_sets:
            merged_set.extend(set.dataset)
            total_num += len(set)

        assert len(merged_set) == total_num

        Trainset = MolDataset(merged_set, self.opt, 'TRAIN')
        self.TrainsetLength = len(Trainset)

        self.trainloader = t.utils.data.DataLoader(Trainset, batch_size = self.batchsize, shuffle = True,
                                                   num_workers = 8, \
                                                   drop_last = True, worker_init_fn = np.random.seed(8),
                                                   pin_memory = True)
        self.validloader = t.utils.data.DataLoader(Validset, batch_size = 1, shuffle = False, num_workers = 0, \
                                                   drop_last = True, worker_init_fn = np.random.seed(8),
                                                   pin_memory = True)
        self.testloader = None

    def BuildOptimizer(self):
        if self.opt.args['Model'] == 'Graphormer':
            self.optimizer = optim.AdamW(self.net.parameters(), lr = 10 ** -self.opt.args['lr'],
                                         weight_decay = 10 ** -self.opt.args['WeightDecay'])
        else:
            self.optimizer = optim.Adam(self.net.parameters(), lr = 10 ** -self.opt.args['lr'],
                                        weight_decay = 10 ** -self.opt.args['WeightDecay'])

    def BuildCriterion(self):
        if self.opt.args['ClassNum'] == 2:
            if self.opt.args['Weight']:
                self.criterion = [nn.CrossEntropyLoss(t.Tensor(weight), reduction = 'mean').\
                                  to(self.device) for weight in self.weights]
            else:
                self.criterion = [nn.CrossEntropyLoss().\
                                      to(self.device) for i in range(self.opt.args['TaskNum'])]
        elif self.opt.args['ClassNum'] == 1:
            self.criterion = [nn.MSELoss().\
                                  to(self.device) for i in range(self.opt.args['TaskNum'])]

    ########################################################

    def SaveModelCkpt(self, ckpt_idx):

        model = self.net
        optimizer = self.optimizer

        addr = self.opt.args['SaveDir'] + 'model/'
        self.CheckDirectory(addr)

        ckpt_name = addr + 'model' + str(ckpt_idx)
        ckpt = {'model': model,
                'optimizer': optimizer}
        t.save(ckpt, ckpt_name)
        print("Model Ckpt Saved!")

    def SaveResultCkpt(self, ckpt_idx, valid_result, test_result=None):
        results = {'cur_opt_valid': valid_result,
                   'cur_opt_test': test_result}
        addr = self.opt.args['SaveDir'] + 'results/'
        self.CheckDirectory(addr)

        result_ckpt_name = addr + 'result' + str(ckpt_idx)
        self.saver.SaveContext(result_ckpt_name, results)
        print('Result saved!')

    def SaveTrainerStatus(self, epoch):
        self.GetControllerStatus(epoch)
        self.controllerstatussaver.SaveStatus(self.status)
        print("Trainer status saved!")

    def GetControllerStatus(self, epoch):
        if self.opt.args['ClassNum'] == 2:
            best_valid = self.BestValidAUC
            test_of_best_valid = self.TestAUCofBestValid


        elif self.opt.args['ClassNum'] == 1:
            best_valid = self.BestValidRMSE
            test_of_best_valid = self.TestRMSEofBestValid


        self.status = {
            'cur_epoch': epoch,
            'best_valid': best_valid,
            'test_of_best_valid': test_of_best_valid
        }

    def CheckDirectory(self, addr):
        if not os.path.exists(addr):
            os.mkdir(addr)

    ########################################################

    def TrainOneOpt(self):
        print("Saving Current opt...")
        self.saver.SaveContext(self.opt.args['SaveDir'] + 'config.json', self.opt.args)

        print("Start Training...")
        epoch = self.StartEpoch
        stop_flag = 0

        if self.opt.args['ClassNum'] == 2:
            self.BestValidAUC = 0
            self.TestAUCofBestValid = 0

        elif self.opt.args['ClassNum'] == 1:
            self.BestValidRMSE = 10e8
            self.TestRMSEofBestValid = 10e8


        while epoch < self.opt.args['MaxEpoch']:
            print('Epoch: ', epoch)

            if stop_flag:
                MaxResult = self.earlystopcontroller.MaxResult
                BestModel = self.earlystopcontroller.MaxResultModelIdx
                print("Early Stop")
                print("The Best Result is: ")
                print(MaxResult)
                print("and its corresponding model ckpt is: ")
                print(BestModel)
                # todo(zqzhang): updated in TPv7
                self.RemoveOtherCkpts(BestModel)
                self.SaveTrainerStopFlag()
                break

            print(f"ESController: {self.earlystopcontroller.MaxResult}")
            print(f"ESController: {self.earlystopcontroller.MaxResultModelIdx}")

            self.TrainOneEpoch(self.net, self.trainloader, self.validloader, self.testloader, self.optimizer, self.criterion, self.evaluator)

            # todo(zqzhang): updated in TPv7
            if not self.no_eval_flag:
                start_time = time.time()
                stop_flag = self.ValidOneTime(epoch, self.net)
                print(f"Eval time: {time.time()-start_time}")
            else:
                self.FakeValidOneTime()
                try:
                    stop_flag = self.ReadStopFlagFile()
                except:
                    stop_flag = False

            self.SaveModelCkpt(epoch)
            self.SaveTrainerStatus(epoch)
            epoch += 1

        MaxResult = self.earlystopcontroller.MaxResult
        BestModel = self.earlystopcontroller.MaxResultModelIdx
        print("Stop Training.")
        print("The Best Result is: ")
        print(MaxResult)
        print("and its corresponding model ckpt is: ")
        print(BestModel)
        # todo(zqzhang): updated in TPv7
        self.RemoveOtherCkpts(BestModel)

        return BestModel, MaxResult

    def TrainOneOpt_KFold(self):
        BaseSaveDir = self.opt.args['SaveDir']
        ckpts = []
        values = []
        for i in range(self.KFold):
            self.opt.set_args('SaveDir', BaseSaveDir + f'{self.KFold}fold-{i}/')
            self.opt.set_args('ModelDir', self.opt.args['SaveDir']+'model/')
            # self.controllerstatussaver.Addr = self.opt.args['SaveDir'] + 'TrainerStatus/'

            self.CheckDirectory(self.opt.args['SaveDir'])
            self.CheckDirectory(self.opt.args['ModelDir'])
            # self.CheckDirectory(self.controllerstatussaver.Addr)

            self.BuildDataset_KFold(i)
            self.reinit()

            ckpt, value = self.TrainOneOpt()
            ckpts.append(ckpt)
            values.append(value)

        return ckpts, values




    ################################################################################################################

    def TrainOneEpoch(self, model, trainloader, validloader, testloader, optimizer, criterion, evaluator):
        cum_loss = 0.0     # cum_loss is used to store the entire loss of a print period for printing the average loss.

        if self.opt.args['PyG']:
            # print("Here b")
            ii = 0
            # print(trainloader)
            for data in trainloader:
                # print(data)
                # print("Here C")
                Label = data.y
                # todo(zqzhang): updated in TPv7
                Label = Label.to(self.device)
                # Label = Label.squeeze(-1)       # [batch, task]
                Label = Label.t()               # [task, batch]
                # print(Label.size())
                # data.to(t.device('cpu'))
                # if data.idx == 101690:
                #     print(f"data:{data}")
                #     print(f"data.x:{data.x}")
                #     print(f"data.edge_index:{data.edge_index}")
                #     print(f"data.edge_attr:{data.edge_attr}")
                #     print(f"data.mask_12:{data.mask_12}")
                #     print(f"data.JT__JT:{data.JT__JT}")
                #     print(f"data.edge_attr_JT:{data.edge_attr_JT}")
                #     print(f"data.atom_num:{data.atom_num}")
                #     print(f"data.bond_num:{data.bond_num}")
                #     print(f"data.singlebond_num:{data.singlebond_num}")
                #     print(f"data.mask_12_length:{data.mask_12_length}")

                output = model(data)
                loss = self.CalculateLoss(output, Label, criterion)
                loss.backward()
                # print("Here D")
            # update the parameters
                if (ii+1) % self.opt.args['UpdateRate'] == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                cum_loss += loss.detach()

            # Print the loss
                if (ii+1) % self.opt.args['PrintRate'] == 0:
                    print("Loss: ", cum_loss.item() / self.opt.args['PrintRate'])
                    cum_loss = 0.0

                # Evaluation
                # todo(zqzhang): updated in TPv7
                if not self.no_eval_flag:
                    if (ii+1) % self.opt.args['ValidRate'] == 0:
                        if self.opt.args['ClassNum'] == 1:
                            print('Running on Valid set')
                            result = self.evaluator.eval(self.validloader, model, [MAE(), RMSE()])
                            if self.testloader:
                                print('Running on Test set')
                                testresult = self.evaluator.eval(self.testloader, model, [MAE(), RMSE()])
                        else:
                            start_time = time.time()
                            print("Running on Valid set")
                            result = self.evaluator.eval(self.validloader, model, [AUC(), ACC()])
                            if self.testloader:
                                print("running on Test set.")
                                testresult = self.evaluator.eval(self.testloader, model, [AUC(), ACC()])
                            print(f"eval time: {time.time()-start_time}")

                ii += 1
        else:
            for ii, data in enumerate(trainloader):
                if self.opt.args['Model'] == 'Graphormer':
                    x, attn_bias, attn_edge_type, spatial_pos, in_degree, out_degree, edge_input, Label, idx = \
                        data.x, data.attn_bias, data.attn_edge_type, data.spatial_pos, \
                        data.in_degree, data.out_degree, data.edge_input, data.y, data.idx
                    Input = {
                        'x': x.to(self.device),
                        'attn_bias': attn_bias.to(self.device),
                        'attn_edge_type': attn_edge_type.to(self.device),
                        'spatial_pos': spatial_pos.to(self.device),
                        'in_degree': in_degree.to(self.device),
                        'out_degree': out_degree.to(self.device),
                        'edge_input': edge_input.to(self.device)
                    }
                    Label = Label.unsqueeze(-1)
                    # print(f"Label: {Label}")

                else:
                    [Input, Label, Idx] = data
                    # todo(zqzhang): updated in TPv7
                    if Input.__class__ == t.Tensor:
                        Input = Input.to(self.device)

                Label = Label.to(self.device)
                Label = Label.squeeze(-1)       # [batch, task]
                Label = Label.t()               # [task, batch]
                # print(Label.size())

                if self.opt.args['Model'] == 'FraGAT':
                    output, _ = model(Input)
                    # raise RuntimeError
                else:
                    output = model(Input)
                # print(f"Output: {output}")
                loss = self.CalculateLoss(output, Label, criterion)
                loss.backward()

            # update the parameters
                if (ii+1) % self.opt.args['UpdateRate'] == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                cum_loss += loss.detach()

            # Print the loss
                if (ii+1) % self.opt.args['PrintRate'] == 0:
                    print("Loss: ", cum_loss.item() / self.opt.args['PrintRate'])
                    cum_loss = 0.0

                # Evaluation
                # todo(zqzhang): updated in TPv7
                if not self.no_eval_flag:
                    if (ii+1) % self.opt.args['ValidRate'] == 0:
                        if self.opt.args['ClassNum'] == 1:
                            print('Running on Valid set')
                            result = self.evaluator.eval(self.validloader, model, [MAE(), RMSE()])
                            if self.testloader:
                                print('Running on Test set')
                                testresult = self.evaluator.eval(self.testloader, model, [MAE(), RMSE()])
                        else:
                            start_time = time.time()
                            print("Running on Valid set")
                            result = self.evaluator.eval(self.validloader, model, [AUC(), ACC()])
                            if self.testloader:
                                print("running on Test set.")
                                testresult = self.evaluator.eval(self.testloader, model, [AUC(), ACC()])
                            print(f"Eval time: {time.time()-start_time}")

        self.lr_sch.step()



    def CalculateLoss(self, output, Label, criterion):
        loss = 0.0
        # print(f"output size: {output.size()}")
        # print(f"label size: {Label.size()}")
        if self.opt.args['ClassNum'] != 1:
            for i in range(self.opt.args['TaskNum']):
                cur_task_output = output[:, i * self.opt.args['ClassNum']: (i + 1) * self.opt.args['ClassNum']]     # select the output of the current task i
                cur_task_label = Label[i]                               # [batch], the label of the current task i
                valid_index = (cur_task_label != -1)                    # Not all of the samples have labels of the current task i.
                valid_label = cur_task_label[valid_index]               # Only the samples that have labels of the current task i will participate in the loss calculation.
                if len(valid_label) == 0:
                    continue
                else:
                    valid_output = cur_task_output[valid_index]
                    loss += criterion[i](valid_output, valid_label)
        else:
            for i in range(self.opt.args['TaskNum']):
                cur_task_output = output[:, i * self.opt.args['ClassNum']: (i + 1) * self.opt.args['ClassNum']]
                cur_task_label = Label[i].unsqueeze(-1)
                loss += criterion[i](cur_task_output, cur_task_label)

        return loss

    ################################################################################################################
    def ValidOneTime(self,epoch,net):
        if self.opt.args['ClassNum'] == 1:
            # todo(zqzhang): updated in TPv7
            if self.opt.args['Splitter'] == None:
                print(f"Evaluate on Training set.")
                trainresult = self.evaluator.eval(self.trainloader, net, [MAE(), RMSE()])

            print('Running on Valid set')
            result = self.evaluator.eval(self.validloader, net, [MAE(), RMSE()])
            if self.testloader:
                print('Running on Test set')
                testresult = self.evaluator.eval(self.testloader, net, [MAE(), RMSE()])

            # todo(zqzhang): updated in TPv7
            if self.opt.args['Splitter'] == None:
                result = trainresult

            valid_result_rmse = result['RMSE']
            if self.testloader:
                test_result_rmse = testresult['RMSE']
            else:
                test_result_rmse = None

            if valid_result_rmse < self.BestValidRMSE:
                self.BestValidRMSE = valid_result_rmse
                self.TestRMSEofBestValid = test_result_rmse
            print('Best Valid: ')
            print(self.BestValidRMSE)
            if self.testloader:
                print('Best Test: ')
                print(self.TestRMSEofBestValid)

            self.SaveResultCkpt(epoch, valid_result_rmse, test_result_rmse)


            if self.testloader:
                stop_flag = self.earlystopcontroller.ShouldStop(result, epoch, testresult)
            else:
                stop_flag = self.earlystopcontroller.ShouldStop(result, epoch)

            return stop_flag

        else:
            # todo(zqzhang): updated in TPv7
            if self.opt.args['Splitter'] == None:
                print(f"Evaluate on Training set.")
                trainresult = self.evaluator.eval(self.trainloader, net, [AUC(), ACC()])

            print("Running on Valid set")
            result = self.evaluator.eval(self.validloader, net, [AUC(), ACC()])
            if self.testloader:
                print("running on Test set.")
                testresult = self.evaluator.eval(self.testloader, net, [AUC(), ACC()])

            # todo(zqzhang): updated in TPv7
            if self.opt.args['Splitter'] == None:
                    result = trainresult

            valid_result_auc = result['AUC']
            if self.testloader:
                test_result_auc = testresult['AUC']
            else:
                test_result_auc = None

            if valid_result_auc > self.BestValidAUC:
                self.BestValidAUC = valid_result_auc
                self.TestAUCofBestValid = test_result_auc
            print('Best Valid: ')
            print(self.BestValidAUC)
            if self.testloader:
                print('Best Test: ')
                print(self.TestAUCofBestValid)

            self.SaveResultCkpt(epoch, valid_result_auc, test_result_auc)

            # todo(zqzhang): updated in TPv7
            if self.opt.args['Splitter'] == None:
                result = trainresult
            if self.testloader:
                stop_flag = self.earlystopcontroller.ShouldStop(result, epoch, testresult)
            else:
                stop_flag = self.earlystopcontroller.ShouldStop(result, epoch)

            return stop_flag

    # todo(zqzhang): updated in TPv7
    def FakeValidOneTime(self):
        for (ii, data) in enumerate(self.validloader):
            a = 1
        for (ii, data) in enumerate(self.testloader):
            a = 1

    # todo(zqzhang): updated in TPv7
    def ReadStopFlagFile(self):
        stop_flag_file = self.opt.args['SaveDir'] + 'stop_flag.json'
        stop_flag = self.saver.LoadContext(stop_flag_file)
        return stop_flag

    def SaveTrainerStopFlag(self):
        trainer_stop_flag_file = self.opt.args['SaveDir'] + 'trainer_stop_flag.json'
        trainer_stop_flag = True
        self.saver.SaveContext(trainer_stop_flag_file, trainer_stop_flag)
        return

    def WeightInit(self):
        for param in self.net.parameters():
            self.initer.WeightInit(param)

    # todo(zqzhang): updated in TPv7
    def RemoveOtherCkpts(self, bestmodel):
        if bestmodel == None:
            print(f"Ckpts will be deleted by Seperated Evaler.")
            return 0

        print(f"Deleting other ckpt models.")
        model_dir = self.opt.args['SaveDir'] + 'model/'
        filenames = os.listdir(model_dir)
        for file in filenames:
            if file != ('model' + str(bestmodel)):
                os.remove(model_dir + file)

        print(f"Deleting other result files.")
        result_dir = self.opt.args['SaveDir'] + 'results/'
        filenames = os.listdir(result_dir)
        for file in filenames:
            if file != ('result' + str(bestmodel)):
                os.remove(result_dir + file)

        print(f"Deleting other TrainerStatus files.")
        status_dir = self.opt.args['SaveDir'] + 'TrainerStatus/'
        filenames = os.listdir(status_dir)
        filename = self.LastFileName(status_dir)
        for file in filenames:
            if file != filename:
                os.remove(status_dir + file)

    def LastFileName(self, Addr):
        dir_files = os.listdir(Addr)
        # print(f"dir_files: {dir_files}")
        # os.listdir returns the file names in Addr, only the names, without the Addr path.
        if dir_files:
            dir_files = sorted(dir_files, key=lambda x: os.path.getctime(os.path.join(Addr, x)))
            last_file = dir_files[-1]
        else:
            last_file = ' '
        # print(f"last_file: {last_file}")
        return last_file

# todo(zqzhang): updated in TPv7
class FinetuneTrainer(object):
    def __init__(self, opt, KFold=False):
        super(FinetuneTrainer, self).__init__()
        self.opt = opt
        self.KFold = KFold
        # self.train_fn = self.TrainOneEpoch

        t.manual_seed(self.opt.args['TorchSeed'])
        #statussaver = ControllerStatusSaver(self.opt.args, 'Trainer')

        self.device = t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')

        self.net = self.BuildModel()
        self.BuildOptimizer()
        self.StartEpoch = 0

        self.BuildDataset()
        self.BuildCriterion()
        self.evaluator = self.BuildEvaluator()
        self.earlystopcontroller = EarlyStopController(self.opt)

        self.saver = Saver()
        self.controllerstatussaver = ControllerStatusSaver(self.opt.args, 'Trainer', restart=True)

    def reinit(self):
        self.net = self.BuildModel()
        self.BuildOptimizer()
        self.StartEpoch = 0

        self.BuildCriterion()
        self.evaluator = self.BuildEvaluator()
        self.earlystopcontroller = EarlyStopController(self.opt)

        self.saver = Saver()
        self.controllerstatussaver = ControllerStatusSaver(self.opt.args, 'Trainer', restart = True)

    ########################################################
    def BuildModel(self):
        net = DNN(
                    self.opt.args['FeatureSize'],
                    self.opt.args['DNNLayers'],
                    self.opt.args['OutputSize'],
                    self.opt
        ).to(self.device)

        return net

    def BuildEvaluator(self):
        evaluator = MLPEvaluator(self.opt)
        return evaluator

    def BuildDataset(self):
        if not self.KFold:
            moldatasetcreator = PretrainMolDatasetCreator(self.opt)
            sets, self.weights = moldatasetcreator.CreateDatasets()

            if len(self.opt.args['SplitRate']) == 2:
                (Trainset, Validset, Testset) = sets
            elif len(self.opt.args['SplitRate']) == 1:
                (Trainset, Validset) = sets
            else:
                (Trainset) = sets

            self.batchsize = self.opt.args['BatchSize']

            assert len(Trainset) >= self.batchsize

            self.trainloader = t.utils.data.DataLoader(Trainset, batch_size = self.batchsize, shuffle = True, num_workers = 8, \
                                              drop_last = True, worker_init_fn = np.random.seed(8), pin_memory = True)
            self.validloader = t.utils.data.DataLoader(Validset, batch_size = 200, shuffle = False, num_workers = 0, \
                                              drop_last = False, worker_init_fn = np.random.seed(8), pin_memory = True)
            if len(self.opt.args['SplitRate']) == 2:
                self.testloader = t.utils.data.DataLoader(Testset, batch_size = 200, shuffle = False, num_workers = 0, \
                                             drop_last = False, worker_init_fn = np.random.seed(8), pin_memory = True)
            else:
                self.testloader = None

        # v3 updated
        if self.KFold:
            split_rate = 1 / self.KFold

            self.splitted_sets = []
            for i in range(self.KFold-1):
                self.opt.set_args('SplitRate',[1-split_rate])
                print(self.opt.args['SplitRate'])
                moldatasetcreator = PretrainMolDatasetCreator(self.opt)
                if i == 0:
                    sets, self.weights = moldatasetcreator.CreateDatasets()
                else:
                    #print(RemainedSet.dataset)
                    sets, _ = moldatasetcreator.CreateDatasets(RemainedSet.dataset)
                (RemainedSet, ValidSet) = sets
                self.splitted_sets.append(ValidSet)
                if i == self.KFold -2:
                    self.splitted_sets.append(RemainedSet)
                split_rate = 1 / (1/split_rate - 1)

            print(f'For {self.KFold}-fold training, the dataset is splitted into:')
            for item in self.splitted_sets:
                print(len(item))

            self.batchsize = self.opt.args['BatchSize']

        #

    def BuildDataset_KFold(self, FoldNum):
        # v3 updated
        # This function is used to extract one subset from the splitted sets as ValidSet, and merge other subsets as TrainSet
        Validset = self.splitted_sets[FoldNum]
        remained_sets = self.splitted_sets.copy()
        remained_sets.remove(Validset)

        # merge
        merged_set = []
        total_num = 0
        for set in remained_sets:
            merged_set.extend(set.dataset)
            total_num += len(set)

        assert len(merged_set) == total_num

        Trainset = PretrainedMolDataset(merged_set, self.opt)

        self.trainloader = t.utils.data.DataLoader(Trainset, batch_size = self.batchsize, shuffle = True,
                                                   num_workers = 8, \
                                                   drop_last = True, worker_init_fn = np.random.seed(8),
                                                   pin_memory = True)
        self.validloader = t.utils.data.DataLoader(Validset, batch_size = 32, shuffle = False, num_workers = 0, \
                                                   drop_last = True, worker_init_fn = np.random.seed(8),
                                                   pin_memory = True)
        self.testloader = None



    def BuildOptimizer(self):
        self.optimizer = optim.Adam(self.net.parameters(), lr = 10 ** -self.opt.args['lr'],
                               weight_decay = 10 ** -self.opt.args['WeightDecay'])

    def BuildCriterion(self):
        if self.opt.args['ClassNum'] == 2:
            if self.opt.args['Weight']:
                self.criterion = [nn.CrossEntropyLoss(t.Tensor(weight), reduction = 'mean').to(self.device) for weight in self.weights]
            else:
                self.criterion = [nn.CrossEntropyLoss().to(self.device) for i in range(self.opt.args['TaskNum'])]
        elif self.opt.args['ClassNum'] == 1:
            self.criterion = [nn.MSELoss().to(self.device) for i in range(self.opt.args['TaskNum'])]

    ########################################################

    def SaveModelCkpt(self, ckpt_idx):

        model = self.net
        optimizer = self.optimizer

        addr = self.opt.args['SaveDir'] + 'model/'
        self.CheckDirectory(addr)

        ckpt_name = addr + 'model' + str(ckpt_idx)
        ckpt = {'model': model,
                'optimizer': optimizer}
        t.save(ckpt, ckpt_name)
        print("Model Ckpt Saved!")

    def SaveResultCkpt(self, ckpt_idx, valid_result, test_result=None):
        results = {'cur_opt_valid': valid_result,
                   'cur_opt_test': test_result}
        addr = self.opt.args['SaveDir'] + 'results/'
        self.CheckDirectory(addr)

        result_ckpt_name = addr + 'result' + str(ckpt_idx)
        self.saver.SaveContext(result_ckpt_name, results)
        print('Result saved!')

    def SaveTrainerStatus(self, epoch):
        self.GetControllerStatus(epoch)
        self.controllerstatussaver.SaveStatus(self.status)
        print("Trainer status saved!")

    def GetControllerStatus(self, epoch):
        if self.opt.args['ClassNum'] == 2:
            best_valid = self.BestValidAUC
            test_of_best_valid = self.TestAUCofBestValid


        elif self.opt.args['ClassNum'] == 1:
            best_valid = self.BestValidRMSE
            test_of_best_valid = self.TestRMSEofBestValid


        self.status = {
            'cur_epoch': epoch,
            'best_valid': best_valid,
            'test_of_best_valid': test_of_best_valid
        }

    def CheckDirectory(self, addr):
        if not os.path.exists(addr):
            os.mkdir(addr)

    ########################################################

    def TrainOneOpt(self):
        print("Saving Current opt...")
        self.saver.SaveContext(self.opt.args['SaveDir'] + 'config.json', self.opt.args)

        print("Start Training...")
        epoch = self.StartEpoch
        stop_flag = 0

        if self.opt.args['ClassNum'] == 2:
            self.BestValidAUC = 0
            self.TestAUCofBestValid = 0

        elif self.opt.args['ClassNum'] == 1:
            self.BestValidRMSE = 10e8
            self.TestRMSEofBestValid = 10e8


        while epoch < self.opt.args['MaxEpoch']:
            print('Epoch: ', epoch)

            if stop_flag:
                MaxResult = self.earlystopcontroller.MaxResult
                BestModel = self.earlystopcontroller.MaxResultModelIdx
                print("Early Stop")
                print("The Best Result is: ")
                print(MaxResult)
                print("and its corresponding model ckpt is: ")
                print(BestModel)
                self.RemoveOtherCkpts(BestModel)
                break


            self.TrainOneEpoch(self.net, self.trainloader, self.validloader, self.testloader, self.optimizer, self.criterion, self.evaluator)
            stop_flag = self.ValidOneTime(epoch, self.net)

            self.SaveModelCkpt(epoch)
            self.SaveTrainerStatus(epoch)
            epoch += 1

        MaxResult = self.earlystopcontroller.MaxResult
        BestModel = self.earlystopcontroller.MaxResultModelIdx
        print("Stop Training.")
        print("The Best Result is: ")
        print(MaxResult)
        print("and its corresponding model ckpt is: ")
        print(BestModel)
        self.RemoveOtherCkpts(BestModel)


        return BestModel, MaxResult

    def TrainOneOpt_KFold(self):
        BaseSaveDir = self.opt.args['SaveDir']
        ckpts = []
        values = []
        for i in range(self.KFold):
            self.opt.set_args('SaveDir', BaseSaveDir + f'{self.KFold}fold-{i}/')
            self.opt.set_args('ModelDir', self.opt.args['SaveDir']+'model/')
            # self.controllerstatussaver.Addr = self.opt.args['SaveDir'] + 'TrainerStatus/'

            self.CheckDirectory(self.opt.args['SaveDir'])
            self.CheckDirectory(self.opt.args['ModelDir'])
            # self.CheckDirectory(self.controllerstatussaver.Addr)

            self.BuildDataset_KFold(i)
            self.reinit()

            ckpt, value = self.TrainOneOpt()
            ckpts.append(ckpt)
            values.append(value)

        return ckpts, values




    ################################################################################################################

    def TrainOneEpoch(self, model, trainloader, validloader, testloader, optimizer, criterion, evaluator):
        cum_loss = 0.0     # cum_loss is used to store the entire loss of a print period for printing the average loss.

        for ii, data in enumerate(trainloader):
            [Input, Label, Idx] = data
            Input = Input.to(self.device)
            # print(Input.device)
            # print(next(model.parameters()).device)
            Label = Label.to(self.device)
            Label = Label.squeeze(-1)       # [batch, task]
            Label = Label.t()               # [task, batch]

            if self.opt.args['Model'] == 'FraGAT':
                output, _ = model(Input)
            else:
                output = model(Input)
            loss = self.CalculateLoss(output, Label, criterion)
            loss.backward()

            # update the parameters
            if (ii+1) % self.opt.args['UpdateRate'] == 0:
                optimizer.step()
                optimizer.zero_grad()

            cum_loss += loss.detach()

            # Print the loss
            if (ii+1) % self.opt.args['PrintRate'] == 0:
                print("Loss: ", cum_loss.item() / self.opt.args['PrintRate'])
                cum_loss = 0.0

            # Evaluation
            if (ii+1) % self.opt.args['ValidRate'] == 0:
                if self.opt.args['ClassNum'] == 1:
                    print('Running on Valid set')
                    result = self.evaluator.eval(self.validloader, model, [MAE(), RMSE()])
                    if self.testloader:
                        print('Running on Test set')
                        testresult = self.evaluator.eval(self.testloader, model, [MAE(), RMSE()])
                else:
                    print("Running on Valid set")
                    result = self.evaluator.eval(self.validloader, model, [AUC(), ACC()])
                    if self.testloader:
                        print("running on Test set.")
                        testresult = self.evaluator.eval(self.testloader, model, [AUC(), ACC()])


    def CalculateLoss(self, output, Label, criterion):
        loss = 0.0
        if self.opt.args['ClassNum'] != 1:
            for i in range(self.opt.args['TaskNum']):
                cur_task_output = output[:, i * self.opt.args['ClassNum']: (i + 1) * self.opt.args['ClassNum']]     # select the output of the current task i
                cur_task_label = Label[i]                               # [batch], the label of the current task i
                valid_index = (cur_task_label != -1)                    # Not all of the samples have labels of the current task i.
                valid_label = cur_task_label[valid_index]               # Only the samples that have labels of the current task i will participate in the loss calculation.
                if len(valid_label) == 0:
                    continue
                else:
                    valid_output = cur_task_output[valid_index]
                    loss += criterion[i](valid_output, valid_label)
        else:
            for i in range(self.opt.args['TaskNum']):
                cur_task_output = output[:, i * self.opt.args['ClassNum']: (i + 1) * self.opt.args['ClassNum']]
                cur_task_label = Label[i].unsqueeze(-1)
                loss += criterion[i](cur_task_output, cur_task_label)

        return loss

    ################################################################################################################
    def ValidOneTime(self,epoch,net):
        if self.opt.args['ClassNum'] == 1:
            print('Running on Valid set')
            result = self.evaluator.eval(self.validloader, net, [MAE(), RMSE()])
            if self.testloader:
                print('Running on Test set')
                testresult = self.evaluator.eval(self.testloader, net, [MAE(), RMSE()])

            valid_result_rmse = result['RMSE']
            if self.testloader:
                test_result_rmse = testresult['RMSE']
            else:
                test_result_rmse = None

            if valid_result_rmse < self.BestValidRMSE:
                self.BestValidRMSE = valid_result_rmse
                self.TestRMSEofBestValid = test_result_rmse
            print('Best Valid: ')
            print(self.BestValidRMSE)
            if self.testloader:
                print('Best Test: ')
                print(self.TestRMSEofBestValid)

            self.SaveResultCkpt(epoch, valid_result_rmse, test_result_rmse)

            if self.testloader:
                stop_flag = self.earlystopcontroller.ShouldStop(result, epoch, testresult)
            else:
                stop_flag = self.earlystopcontroller.ShouldStop(result, epoch)

            return stop_flag

        else:
            print("Running on Valid set")


            result = self.evaluator.eval(self.validloader, net, [AUC(), ACC()])
            if self.testloader:
                print("running on Test set.")
                testresult = self.evaluator.eval(self.testloader, net, [AUC(), ACC()])

            valid_result_auc = result['AUC']
            if self.testloader:
                test_result_auc = testresult['AUC']
            else:
                test_result_auc = None

            if valid_result_auc > self.BestValidAUC:
                self.BestValidAUC = valid_result_auc
                self.TestAUCofBestValid = test_result_auc
            print('Best Valid: ')
            print(self.BestValidAUC)
            if self.testloader:
                print('Best Test: ')
                print(self.TestAUCofBestValid)

            self.SaveResultCkpt(epoch, valid_result_auc, test_result_auc)

            if self.testloader:
                stop_flag = self.earlystopcontroller.ShouldStop(result, epoch, testresult)
            else:
                stop_flag = self.earlystopcontroller.ShouldStop(result, epoch)

            return stop_flag

    def RemoveOtherCkpts(self, bestmodel):
        if bestmodel == None:
            print(f"Ckpts will be deleted by Seperated Evaler.")
            return 0

        print(f"Deleting other ckpt models.")
        model_dir = self.opt.args['SaveDir'] + 'model/'
        filenames = os.listdir(model_dir)
        for file in filenames:
            if file != ('model' + str(bestmodel)):
                os.remove(model_dir + file)

        print(f"Deleting other result files.")
        result_dir = self.opt.args['SaveDir'] + 'results/'
        filenames = os.listdir(result_dir)
        for file in filenames:
            if file != ('result' + str(bestmodel)):
                os.remove(result_dir + file)

        print(f"Deleting other TrainerStatus files.")
        status_dir = self.opt.args['SaveDir'] + 'TrainerStatus/'
        filenames = os.listdir(status_dir)
        filename = self.LastFileName(status_dir)
        for file in filenames:
            if file != filename:
                os.remove(status_dir + file)

    def LastFileName(self, Addr):
        dir_files = os.listdir(Addr)
        # print(f"dir_files: {dir_files}")
        # os.listdir returns the file names in Addr, only the names, without the Addr path.
        if dir_files:
            dir_files = sorted(dir_files, key=lambda x: os.path.getctime(os.path.join(Addr, x)))
            last_file = dir_files[-1]
        else:
            last_file = ' '
        # print(f"last_file: {last_file}")
        return last_file

class ToysetTrainer(object):
    def __init__(self, opt, KFold=False):
        super(ToysetTrainer, self).__init__()
        self.opt = opt
        self.KFold = KFold
        # self.train_fn = self.TrainOneEpoch

        t.manual_seed(self.opt.args['TorchSeed'])
        #statussaver = ControllerStatusSaver(self.opt.args, 'Trainer')

        self.device = t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')

        self.net = self.BuildModel()
        self.BuildOptimizer()
        self.StartEpoch = 0

        self.BuildDataset()
        self.BuildCriterion()
        self.evaluator = self.BuildEvaluator()
        self.earlystopcontroller = EarlyStopController(self.opt)

        self.saver = Saver()
        self.controllerstatussaver = ControllerStatusSaver(self.opt.args, 'Trainer', restart=True)

    def reinit(self):
        self.net = self.BuildModel()
        self.BuildOptimizer()
        self.StartEpoch = 0

        self.BuildCriterion()
        self.evaluator = self.BuildEvaluator()
        self.earlystopcontroller = EarlyStopController(self.opt)

        self.saver = Saver()
        self.controllerstatussaver = ControllerStatusSaver(self.opt.args, 'Trainer', restart = True)

    ########################################################
    def BuildModel(self):
        if self.opt.args['Model'] == 'MLP':
            net = DNN(
                    self.opt.args['FeatureSize'],
                    self.opt.args['DNNLayers'],
                    self.opt.args['OutputSize'],
                    self.opt
            ).to(self.device)
        else:
            raise RuntimeError(f"Given wrong model choice.")
        return net

    def BuildEvaluator(self):
        evaluator = MLPEvaluator(self.opt)
        return evaluator

    def BuildDataset(self):
        if not self.KFold:
            moldatasetcreator = PretrainMolDatasetCreator(self.opt)
            sets, self.weights = moldatasetcreator.CreateDatasets()

            if len(self.opt.args['SplitRate']) == 2:
                (Trainset, Validset, Testset) = sets
            elif len(self.opt.args['SplitRate']) == 1:
                (Trainset, Validset) = sets
            else:
                (Trainset) = sets

            self.batchsize = self.opt.args['BatchSize']

            assert len(Trainset) >= self.batchsize

            self.trainloader = t.utils.data.DataLoader(Trainset, batch_size = self.batchsize, shuffle = True, num_workers = 0, \
                                              drop_last = True, worker_init_fn = np.random.seed(8), pin_memory = True)
            self.validloader = t.utils.data.DataLoader(Validset, batch_size = 200, shuffle = False, num_workers = 0, \
                                              drop_last = False, worker_init_fn = np.random.seed(8), pin_memory = True)
            if len(self.opt.args['SplitRate']) == 2:
                self.testloader = t.utils.data.DataLoader(Testset, batch_size = 200, shuffle = False, num_workers = 0, \
                                             drop_last = False, worker_init_fn = np.random.seed(8), pin_memory = True)
            else:
                self.testloader = None

        # v3 updated
        if self.KFold:
            split_rate = 1 / self.KFold

            self.splitted_sets = []
            for i in range(self.KFold-1):
                self.opt.set_args('SplitRate',[1-split_rate])
                print(self.opt.args['SplitRate'])
                moldatasetcreator = PretrainMolDatasetCreator(self.opt)
                if i == 0:
                    sets, self.weights = moldatasetcreator.CreateDatasets()
                else:
                    #print(RemainedSet.dataset)
                    sets, _ = moldatasetcreator.CreateDatasets(RemainedSet.dataset)
                (RemainedSet, ValidSet) = sets
                self.splitted_sets.append(ValidSet)
                if i == self.KFold -2:
                    self.splitted_sets.append(RemainedSet)
                split_rate = 1 / (1/split_rate - 1)

            print(f'For {self.KFold}-fold training, the dataset is splitted into:')
            for item in self.splitted_sets:
                print(len(item))

            self.batchsize = self.opt.args['BatchSize']

        #

    def BuildDataset_KFold(self, FoldNum):
        # v3 updated
        # This function is used to extract one subset from the splitted sets as ValidSet, and merge other subsets as TrainSet
        Validset = self.splitted_sets[FoldNum]
        remained_sets = self.splitted_sets.copy()
        remained_sets.remove(Validset)

        # merge
        merged_set = []
        total_num = 0
        for set in remained_sets:
            merged_set.extend(set.dataset)
            total_num += len(set)

        assert len(merged_set) == total_num

        Trainset = PretrainedMolDataset(merged_set, self.opt)

        self.trainloader = t.utils.data.DataLoader(Trainset, batch_size = self.batchsize, shuffle = True,
                                                   num_workers = 8, \
                                                   drop_last = True, worker_init_fn = np.random.seed(8),
                                                   pin_memory = True)
        self.validloader = t.utils.data.DataLoader(Validset, batch_size = 32, shuffle = False, num_workers = 0, \
                                                   drop_last = True, worker_init_fn = np.random.seed(8),
                                                   pin_memory = True)
        self.testloader = None



    def BuildOptimizer(self):
        self.optimizer = optim.Adam(self.net.parameters(), lr = 10 ** -self.opt.args['lr'],
                               weight_decay = 10 ** -self.opt.args['WeightDecay'])

    def BuildCriterion(self):
        if self.opt.args['ClassNum'] == 2:
            if self.opt.args['Weight']:
                self.criterion = [nn.CrossEntropyLoss(t.Tensor(weight), reduction = 'mean').to(self.device) for weight in self.weights]
            else:
                self.criterion = [nn.CrossEntropyLoss().to(self.device) for i in range(self.opt.args['TaskNum'])]
        elif self.opt.args['ClassNum'] == 1:
            self.criterion = [nn.MSELoss().to(self.device) for i in range(self.opt.args['TaskNum'])]

    ########################################################

    def SaveModelCkpt(self, ckpt_idx):

        model = self.net
        optimizer = self.optimizer

        addr = self.opt.args['SaveDir'] + 'model/'
        self.CheckDirectory(addr)

        ckpt_name = addr + 'model' + str(ckpt_idx)
        ckpt = {'model': model,
                'optimizer': optimizer}
        t.save(ckpt, ckpt_name)
        print("Model Ckpt Saved!")

    def SaveResultCkpt(self, ckpt_idx, valid_result, test_result=None):
        results = {'cur_opt_valid': valid_result,
                   'cur_opt_test': test_result}
        addr = self.opt.args['SaveDir'] + 'results/'
        self.CheckDirectory(addr)

        result_ckpt_name = addr + 'result' + str(ckpt_idx)
        self.saver.SaveContext(result_ckpt_name, results)
        print('Result saved!')

    def SaveTrainerStatus(self, epoch):
        self.GetControllerStatus(epoch)
        self.controllerstatussaver.SaveStatus(self.status)
        print("Trainer status saved!")

    def GetControllerStatus(self, epoch):
        if self.opt.args['ClassNum'] == 2:
            best_valid = self.BestValidAUC
            test_of_best_valid = self.TestAUCofBestValid


        elif self.opt.args['ClassNum'] == 1:
            best_valid = self.BestValidRMSE
            test_of_best_valid = self.TestRMSEofBestValid


        self.status = {
            'cur_epoch': epoch,
            'best_valid': best_valid,
            'test_of_best_valid': test_of_best_valid
        }

    def CheckDirectory(self, addr):
        if not os.path.exists(addr):
            os.mkdir(addr)

    ########################################################

    def TrainOneOpt(self):
        print("Saving Current opt...")
        self.saver.SaveContext(self.opt.args['SaveDir'] + 'config.json', self.opt.args)

        print("Start Training...")
        epoch = self.StartEpoch
        stop_flag = 0

        if self.opt.args['ClassNum'] == 2:
            self.BestValidAUC = 0
            self.TestAUCofBestValid = 0

        elif self.opt.args['ClassNum'] == 1:
            self.BestValidRMSE = 10e8
            self.TestRMSEofBestValid = 10e8


        while epoch < self.opt.args['MaxEpoch']:
            print('Epoch: ', epoch)

            if stop_flag:
                MaxResult = self.earlystopcontroller.MaxResult
                BestModel = self.earlystopcontroller.MaxResultModelIdx
                print("Early Stop")
                print("The Best Result is: ")
                print(MaxResult)
                print("and its corresponding model ckpt is: ")
                print(BestModel)
                self.RemoveOtherCkpts(BestModel)
                break


            self.TrainOneEpoch(self.net, self.trainloader, self.validloader, self.testloader, self.optimizer, self.criterion, self.evaluator)
            stop_flag = self.ValidOneTime(epoch, self.net)

            self.SaveModelCkpt(epoch)
            self.SaveTrainerStatus(epoch)
            epoch += 1

        MaxResult = self.earlystopcontroller.MaxResult
        BestModel = self.earlystopcontroller.MaxResultModelIdx
        print("Stop Training.")
        print("The Best Result is: ")
        print(MaxResult)
        print("and its corresponding model ckpt is: ")
        print(BestModel)
        self.RemoveOtherCkpts(BestModel)


        return BestModel, MaxResult

    def TrainOneOpt_KFold(self):
        BaseSaveDir = self.opt.args['SaveDir']
        ckpts = []
        values = []
        for i in range(self.KFold):
            self.opt.set_args('SaveDir', BaseSaveDir + f'{self.KFold}fold-{i}/')
            self.opt.set_args('ModelDir', self.opt.args['SaveDir']+'model/')
            # self.controllerstatussaver.Addr = self.opt.args['SaveDir'] + 'TrainerStatus/'

            self.CheckDirectory(self.opt.args['SaveDir'])
            self.CheckDirectory(self.opt.args['ModelDir'])
            # self.CheckDirectory(self.controllerstatussaver.Addr)

            self.BuildDataset_KFold(i)
            self.reinit()

            ckpt, value = self.TrainOneOpt()
            ckpts.append(ckpt)
            values.append(value)

        return ckpts, values




    ################################################################################################################

    def TrainOneEpoch(self, model, trainloader, validloader, testloader, optimizer, criterion, evaluator):
        cum_loss = 0.0     # cum_loss is used to store the entire loss of a print period for printing the average loss.

        for ii, data in enumerate(trainloader):
            [Input, Label, Idx] = data
            Input = Input.to(self.device)
            # print(Input.device)
            # print(next(model.parameters()).device)
            Label = Label.to(self.device)
            Label = Label.squeeze(-1)       # [batch, task]
            Label = Label.t()               # [task, batch]

            if self.opt.args['Model'] == 'FraGAT':
                output, _ = model(Input)
            else:
                output = model(Input)
            loss = self.CalculateLoss(output, Label, criterion)
            loss.backward()

            # update the parameters
            if (ii+1) % self.opt.args['UpdateRate'] == 0:
                optimizer.step()
                optimizer.zero_grad()

            cum_loss += loss.detach()

            # Print the loss
            if (ii+1) % self.opt.args['PrintRate'] == 0:
                print("Loss: ", cum_loss.item() / self.opt.args['PrintRate'])
                cum_loss = 0.0

            # Evaluation
            if (ii+1) % self.opt.args['ValidRate'] == 0:
                if self.opt.args['ClassNum'] == 1:
                    print('Running on Valid set')
                    result = self.evaluator.eval(self.validloader, model, [MAE(), RMSE()])
                    if self.testloader:
                        print('Running on Test set')
                        testresult = self.evaluator.eval(self.testloader, model, [MAE(), RMSE()])
                else:
                    print("Running on Valid set")
                    result = self.evaluator.eval(self.validloader, model, [AUC(), ACC()])
                    if self.testloader:
                        print("running on Test set.")
                        testresult = self.evaluator.eval(self.testloader, model, [AUC(), ACC()])


    def CalculateLoss(self, output, Label, criterion):
        loss = 0.0
        if self.opt.args['ClassNum'] != 1:
            for i in range(self.opt.args['TaskNum']):
                cur_task_output = output[:, i * self.opt.args['ClassNum']: (i + 1) * self.opt.args['ClassNum']]     # select the output of the current task i
                cur_task_label = Label[i]                               # [batch], the label of the current task i
                valid_index = (cur_task_label != -1)                    # Not all of the samples have labels of the current task i.
                valid_label = cur_task_label[valid_index]               # Only the samples that have labels of the current task i will participate in the loss calculation.
                if len(valid_label) == 0:
                    continue
                else:
                    valid_output = cur_task_output[valid_index]
                    loss += criterion[i](valid_output, valid_label)
        else:
            for i in range(self.opt.args['TaskNum']):
                cur_task_output = output[:, i * self.opt.args['ClassNum']: (i + 1) * self.opt.args['ClassNum']]
                cur_task_label = Label[i].unsqueeze(-1)
                loss += criterion[i](cur_task_output, cur_task_label)

        return loss

    ################################################################################################################
    def ValidOneTime(self,epoch,net):
        if self.opt.args['ClassNum'] == 1:
            print('Running on Valid set')
            result = self.evaluator.eval(self.validloader, net, [MAE(), RMSE()])
            if self.testloader:
                print('Running on Test set')
                testresult = self.evaluator.eval(self.testloader, net, [MAE(), RMSE()])

            valid_result_rmse = result['RMSE']
            if self.testloader:
                test_result_rmse = testresult['RMSE']
            else:
                test_result_rmse = None

            if valid_result_rmse < self.BestValidRMSE:
                self.BestValidRMSE = valid_result_rmse
                self.TestRMSEofBestValid = test_result_rmse
            print('Best Valid: ')
            print(self.BestValidRMSE)
            if self.testloader:
                print('Best Test: ')
                print(self.TestRMSEofBestValid)

            self.SaveResultCkpt(epoch, valid_result_rmse, test_result_rmse)

            if self.testloader:
                stop_flag = self.earlystopcontroller.ShouldStop(result, epoch, testresult)
            else:
                stop_flag = self.earlystopcontroller.ShouldStop(result, epoch)

            return stop_flag

        else:
            print("Running on Valid set")


            result = self.evaluator.eval(self.validloader, net, [AUC(), ACC()])
            if self.testloader:
                print("running on Test set.")
                testresult = self.evaluator.eval(self.testloader, net, [AUC(), ACC()])

            valid_result_auc = result['AUC']
            if self.testloader:
                test_result_auc = testresult['AUC']
            else:
                test_result_auc = None

            if valid_result_auc > self.BestValidAUC:
                self.BestValidAUC = valid_result_auc
                self.TestAUCofBestValid = test_result_auc
            print('Best Valid: ')
            print(self.BestValidAUC)
            if self.testloader:
                print('Best Test: ')
                print(self.TestAUCofBestValid)

            self.SaveResultCkpt(epoch, valid_result_auc, test_result_auc)

            if self.testloader:
                stop_flag = self.earlystopcontroller.ShouldStop(result, epoch, testresult)
            else:
                stop_flag = self.earlystopcontroller.ShouldStop(result, epoch)

            return stop_flag

    def RemoveOtherCkpts(self, bestmodel):
        if bestmodel == None:
            print(f"Ckpts will be deleted by Seperated Evaler.")
            return 0

        print(f"Deleting other ckpt models.")
        model_dir = self.opt.args['SaveDir'] + 'model/'
        filenames = os.listdir(model_dir)
        for file in filenames:
            if file != ('model' + str(bestmodel)):
                os.remove(model_dir + file)

        print(f"Deleting other result files.")
        result_dir = self.opt.args['SaveDir'] + 'results/'
        filenames = os.listdir(result_dir)
        for file in filenames:
            if file != ('result' + str(bestmodel)):
                os.remove(result_dir + file)

        print(f"Deleting other TrainerStatus files.")
        status_dir = self.opt.args['SaveDir'] + 'TrainerStatus/'
        filenames = os.listdir(status_dir)
        filename = self.LastFileName(status_dir)
        for file in filenames:
            if file != filename:
                os.remove(status_dir + file)

    def LastFileName(self, Addr):
        dir_files = os.listdir(Addr)
        # print(f"dir_files: {dir_files}")
        # os.listdir returns the file names in Addr, only the names, without the Addr path.
        if dir_files:
            dir_files = sorted(dir_files, key=lambda x: os.path.getctime(os.path.join(Addr, x)))
            last_file = dir_files[-1]
        else:
            last_file = ' '
        # print(f"last_file: {last_file}")
        return last_file

################################################
# Test codes
################################################
# Config Controller Test
# Path Illustration:
# RootPath : ./RootPath/                                 the path of the entire Experiment of different datasets
# TrialPath: ./RootPath/ExpName/                         the path of the experiment of the current dataset
# ExpDir: ./RootPath/ExpName/expi                        the path of the experiment of the current opt
# SaveDir:./RootPath/ExpName/expi/j/                     the path of the experiment of the current torch seed (run multi times of the same opt)
# ModelDir: ./RootPath/ExpName/expi/i/model/             the path to save the ckpts.
'''
BasicParamList = {
    'ExpName': 'ExpName',
    'MainMetric': 'RMSE',
    'DataPath': './data/ExpName_SMILESValue.txt',
    'RootPath': './RootPath/',
    'CUDA_VISIBLE_DEVICES': '0',
    'TaskNum': 1,
    'ClassNum': 1,
    'Weight': True,
    'ValidRate': 4000,
    'PrintRate': 20,
    'Frag': True,
    'output_size': 1,
    'atom_feature_size': 39,
    'bond_feature_size': 10,
    'Feature': 'AttentiveFP',
    'ValidBalance': False,
    'TestBalance': False,
    'MaxEpoch': 800,
    'SplitRate': [0.8],
    'Splitter': 'Random',
    'UpdateRate': 1,
    'LowerThanMaxLimit': 3,
    'DecreasingLimit': 2
}
AdjustableParamList = {
    'SplitValidSeed': [8, 28, 58, 88],
    'SplitTestSeed': [8, 28, 58, 88, 108],
    'FP_size': [32, 64, 128],
    'atom_layers': [2, 3],
}
SpecificParamList = {
    'SplitValidSeed': [8, 28],
    'SplitTestSeed': [8, 28],
    'FP_size': [150, 150],
    'atom_layers': [3, 3],
}

configcontroller = GreedyConfigController(BasicParamList, AdjustableParamList, SpecificParamList)
print('EXP 0:')
opt = configcontroller.GetOpts()
print("All params should be initial.")
print("For the test params, they should be: ValidSeed = 8, TestSeed = 8, FP_size = 150, atom_layers = 3.")
print(opt.args)
configcontroller.StoreResults(0)

print("Before calling AdjustParams(), the opts are not changed.")
configcontroller.AdjustParams()
print("After calling AdjustParams(), the opts are changed.")

print('EXP 1:')
opt = configcontroller.GetOpts()
print("One step adjust is done. All params should be the next one.")
print("For the test params, they should be: ValidSeed=28, SplitSeed = 28, FP_size = 150, atom_layers = 3.")
print(opt.args)
configcontroller.StoreResults(1)
configcontroller.AdjustParams()

print("Start adjusting params by greedy search.")
print("In the following 8 opts, the params shoud be:")
print("8 8 32 2")
print("28 8 32 2")
print("58 8 32 2")
print("88 8 32 2")
print("8 28 32 2")
print("8 58 32 2")
print("8 88 32 2")
print("8 108 32 2")
for i in range(8):
    print('EXP ', i+2, ':')
    opt = configcontroller.GetOpts()
    print(opt.args)
    configcontroller.StoreResults(i)
    configcontroller.AdjustParams()
    print(configcontroller.GetControllerStatus()['result'])

print("Although the opt of EXP 9 is 8 108 32 2, the status of the configcontroller have been modified when calling AdjustParams()")
print("So the opt in the current status is the next opt: 8 8 64 2.")
print("Current status of the ConfigController: ")
status = configcontroller.GetControllerStatus()
print(status)
print("Current status have been saved.")
print("Init a new controller and set the status.")
configcontrollernew = GreedyConfigController(BasicParamList, AdjustableParamList, SpecificParamList)
configcontrollernew.SetControllerStatus(status)

print('EXP 10:')
print("The opt of exp 10 should be: 8 8 64 2.")
opt = configcontrollernew.GetOpts()
print(opt.args)
configcontrollernew.StoreResults(9)
configcontrollernew.AdjustParams()
print(configcontrollernew.GetControllerStatus()['result'])

#####################################################################################################
# Early Stop Controller Test
ESP = EarlyStopController(opt)
score = [{'RMSE': 3},
         {'RMSE': 5},
         {'RMSE': 4},
         {'RMSE': 2},
         {'RMSE': 2.5},
         {'RMSE': 2.6},
         {'RMSE': 2.4},
         {'RMSE': 7},
         {'RMSE': 9},
         {'RMSE': 11}
         ]
testscore = [{'RMSE': 1},{'RMSE': 2},{'RMSE': 3},{'RMSE': 4},{'RMSE': 5},{'RMSE': 6},{'RMSE': 7},{'RMSE': 8},{'RMSE': 9},{'RMSE': 10}]
for i in range(8):
    end_flag = ESP.ShouldStop(score[i],i,testscore[i])
    print(end_flag)
    if end_flag==True:
        print(ESP.GetControllerStatus())
print(ESP.BestModel())
status = ESP.GetControllerStatus()
ESPnew = EarlyStopController(opt)
ESPnew.SetControllerStatus(status)
for i in range(2):
    end_flag = ESPnew.ShouldStop(score[i+8],i+8,testscore[i+8])
    print(end_flag)
    print(ESPnew.GetControllerStatus())
###########################################################################################################
# CkptController Test
opt.set_args('SaveDir', opt.args['ExpDir'] + '1/')
os.mkdir(opt.args['SaveDir'])
opt.set_args('ModelDir', opt.args['SaveDir'] + 'model/')
os.mkdir(opt.args['ModelDir'])
ckptcontroller = CkptController(opt)
model = t.nn.Linear(5,10)
optimizer = t.optim.Adam(model.parameters(), lr=0.1)
for i in range(9):
    scores = {'RMSE': i}
    testscores = {'RMSE': i}
    ckptcontroller.CkptProcessing(model, optimizer, i, scores, testscores)
status = ckptcontroller.GetControllerStatus()
print(status)
ckptcontrollernew = CkptController(opt)
ckptcontrollernew.SetControllerStatus(status)
i = 9
scores = {'RMSE': i}
testscores = {'RMSE': i}
ckptcontrollernew.CkptProcessing(model, optimizer, i, scores, testscores)
modelnew, optimizernew, epochnew = ckptcontrollernew.LoadCkpt()
print(modelnew)
print(optimizernew)
print(epochnew)
print(ckptcontrollernew.GetControllerStatus())
#############################################################################################################
'''
