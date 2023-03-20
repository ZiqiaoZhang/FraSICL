from TrainingFramework.ProcessControllers import Controller, GreedyConfigController, EarlyStopController, Saver, ControllerStatusSaver, OnlyEvaler, SeperatedEvaler
import torch as t
import random
import os
import re
from TrainingFramework.Dataset import MolDatasetCreator, MolDataset, PretrainMolDatasetCreator, PretrainedMolDataset, ToysetCreator
from TrainingFramework.FileUtils import *
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

from ContrastiveComponent.ContrastiveModel import FraContra
from ContrastiveComponent.ContrastiveFeaturizer import PyGFraGATContraFeaturizer
from ContrastiveComponent.Utils import NTXentLoss
from ContrastiveComponent.ContrastiveEvaluator import PyGFraGATContraFinetuneEvaluator

import torch_geometric as tg

class PretrainExperimentProcessController(Controller):
    # Module to control the entire experimental process.
    def __init__(self, ExpOptions, Params):
        # ExpOptions: Options to set the ExperimentProcessController
        # Params: Params for the experiment. i.g. the three ParamLists of greedy search.
        super(PretrainExperimentProcessController, self).__init__()

        self.ExpOptions = ExpOptions
        self.search = self.ExpOptions['Search']
        self.seedperopt = self.ExpOptions['SeedPerOpt']
        self.PretrainFinetuneMode = self.ExpOptions['PretrainFinetune']
        self.TorchSeedBias = self.ExpOptions['TorchSeedBias']

        # process the params based on different searching methods, determined by the ExpOptions
        if self.search == 'greedy':
            self.BasicParamList, self.AdjustableParamList, self.SpecificParamList = Params


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

        if self.PretrainFinetuneMode == 'Pretrain':
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

                # if self.Finetune:
                #     trainer = FinetuneTrainer(opt)
                # else:
                if no_eval_flag:
                    trainer = PreTrainer(opt,no_eval_flag)
                    avg_loss = trainer.TrainOneOpt()
                    print(f"cur_opt_cur_seed_value:{avg_loss}")
                    self.cur_opt_results.append(avg_loss)

                else:
                    trainer = FinetuneTrainer(opt)
                    ckpt, value = trainer.TrainOneOpt()
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

class PreTrainer(object):
    def __init__(self, opt, no_eval_flag=False):
        super(PreTrainer, self).__init__()
        self.opt = opt
        # self.train_fn = self.TrainOneEpoch
        self.device = t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')
        # self.device = t.device('cpu')

        self.no_eval_flag = no_eval_flag

        t.manual_seed(self.opt.args['TorchSeed'])

        np.random.seed(self.opt.args['TorchSeed'])
        random.seed(self.opt.args['TorchSeed'])
        os.environ['PYTHONHASHSEED'] = str(self.opt.args['TorchSeed'])
        t.cuda.manual_seed(self.opt.args['TorchSeed'])
        t.cuda.manual_seed_all(self.opt.args['TorchSeed'])

        self.BuildPretrainDataset()

        self.net = self.BuildModel()
        self.BuildIniter()
        if self.initer:
            self.WeightInit()

        self.BuildPretrainOptimizer()

        self.StartEpoch = 0

        self.lr_sch = self.BuildScheduler()
        self.BuildPretrainCriterion()

        self.start_ii = 0

        self.saver = Saver()
        self.controllerstatussaver = ControllerStatusSaver(self.opt.args, 'Trainer', restart=True)


        # self.LoadModelCkpt()

    ##########################################
    def BuildModel(self):
        net = FraContra(self.opt).to(self.device)
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


    def BuildPretrainDataset(self):
        # Load dataset
        fileloader = FileLoader(self.opt.args['PretrainDataPath'])
        dataset = fileloader.load()
        total_size = len(dataset)
        self.batchsize = self.opt.args['BatchSize']


        # split dataset into Train and Valid randomly
        print(f"Split Pretrain dataset into Train and Valid...")
        split_rate = self.opt.args['PretrainValidSplitRate']
        # random
        seed = self.opt.args['PretrainDatasetValidSeed']
        random.seed(seed)
        random.shuffle(dataset)
        # split
        Trainset = dataset[:int(total_size*split_rate)]
        Validset = dataset[int(total_size*split_rate):]

        # Split the Trainset into parts, and pre_compute their Tensors
        if self.opt.args['SplittingTrainset']:
            random.shuffle(Trainset)
            one_part_length = int(len(Trainset)/self.opt.args['SplittingTrainset'])
            self.pretrainloaders = []
            for i in range(self.opt.args['SplittingTrainset']):
                if i  < self.opt.args['SplittingTrainset']-1:
                    Trainset_partial = Trainset[i*one_part_length : (i+1)*one_part_length]
                else:
                    Trainset_partial = Trainset[(i)*one_part_length:]
                    # Maybe wrong
                    # i = 0,1,2,3,4,5,6,7,8 => 0:19000, ..., 19000*8:19000*9
                    # then, i=9 => 8*19000:19000*10???
                    # So, the last part may be wrong.
                    # should check whether the first half of part 9 is the same with part 8
                    # Then, delete the first half of part 9.
                print(f"Length of Trainset part {i}: {len(Trainset_partial)}")
                TrainsetPartial = MolDataset(Trainset_partial, self.opt, 'TRAIN', i)
                self.pretrainloaders.append(
                        tg.loader.DataLoader(TrainsetPartial, batch_size = self.batchsize, shuffle = True, num_workers = 8,\
                                                   drop_last = True, worker_init_fn = np.random.seed(8), pin_memory = True)
                )
        else:
        # Build dataset object and dataloader object
            PretrainDataset = MolDataset(Trainset, self.opt, 'TRAIN')
            self.TrainsetLength = len(PretrainDataset)
            print(f"Total size of the Pretrain dataset is:{self.TrainsetLength}")
            self.pretrainloader = tg.loader.DataLoader(PretrainDataset, batch_size = self.batchsize, shuffle = True,
                                                   num_workers = 2, \
                                                   drop_last = True, worker_init_fn = np.random.seed(8),
                                                   pin_memory = True)

        ValidDataset = MolDataset(Validset, self.opt, 'EVAL')
        self.ValidsetLength = len(ValidDataset)
        self.pretrainevalloader = tg.loader.DataLoader(ValidDataset, batch_size = self.batchsize, shuffle = False, num_workers = 2,\
                                                   drop_last = True, worker_init_fn = np.random.seed(8), pin_memory = True)


    def BuildPretrainOptimizer(self):
        self.optimizer = optim.Adam(self.net.parameters(), lr = 10 ** -self.opt.args['lr'],
                                        weight_decay = 10 ** -self.opt.args['WeightDecay'])

    def BuildPretrainCriterion(self):
        self.L2Loss = nn.MSELoss().to(self.device)
        self.NTXentLoss = NTXentLoss(self.opt)


    ########################################################

    def SaveModelCkpt(self, ckpt_idx, avg_loss):
        model = self.net
        optimizer = self.optimizer
        scheduler = self.lr_sch

        addr = self.opt.args['SaveDir'] + 'model/'
        self.CheckDirectory(addr)

        ckpt_name = addr + 'model-' + (ckpt_idx)
        ckpt = {'model': model,
                'optimizer': optimizer,
                'scheduler': scheduler,
                }
        t.save(ckpt, ckpt_name)

        avg_loss_filename = addr + 'avg_loss-' + (ckpt_idx)
        self.saver.SaveContext(avg_loss_filename, avg_loss.item())
        print("Model Ckpt Saved!")

    def LoadModelCkpt(self):
        addr = self.opt.args['SaveDir'] + 'model/'
        self.CheckDirectory(addr)

        ckpt_name = self.LastFileName(addr)
        if ckpt_name != ' ':
            print(f"Load Model ckpt")
            ckpt_file_path = addr+ckpt_name
            ckpt = t.load(ckpt_file_path)

            self.net = ckpt['model']
            self.optimizer = ckpt['optimizer']
            self.lr_sch = ckpt['scheduler']
            print(f"Model ckpt loaded!")

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


        self.status = {
            'cur_epoch': epoch,
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

        while epoch < self.opt.args['MaxEpoch']:
            print('Epoch: ', epoch)
            self.TrainOneEpoch(self.net, epoch)

            print(f"Evaluating Model Loss on Valid Set...")
            avg_loss = self.ValidOneTime(self.net)
            print(f"Saving Model Checkpoint {epoch}-fin...")
            self.SaveModelCkpt(f"{epoch}-fin", avg_loss)
            # self.SaveTrainerStatus(epoch)
            epoch += 1


        return avg_loss


    ################################################################################################################

    def TrainOneEpoch(self, model, epoch):
        cum_loss = 0.0     # cum_loss is used to store the entire loss of a print period for printing the average loss.
        save_cnt = 0
        if self.opt.args['PyG']:
            ii = 0
            if self.opt.args['SplittingTrainset']:
                random.shuffle(self.pretrainloaders)
                for trainloader in self.pretrainloaders:

                    for _, data in enumerate(trainloader):
                        print(f"iter: {ii}")
                        if ii < self.start_ii:
                            continue
                        MolEmbeddings, FragViewEmbeddings, FragEmbedsSim = model(data)

                        loss = self.CalculateLoss(MolEmbeddings, FragViewEmbeddings, FragEmbedsSim,
                                                  data[0].singlebond_num)
                        loss.backward()
                        # update the parameters
                        if (ii + 1) % self.opt.args['UpdateRate'] == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()

                        cum_loss += loss.detach()

                        # Print the loss
                        if (ii + 1) % self.opt.args['PrintRate'] == 0:
                            print("Loss: ", cum_loss.item() / self.opt.args['PrintRate'])
                            cum_loss = 0.0

                        if (ii + 1) % self.opt.args['SaveCkptRate'] == 0:
                            # save ckpt
                            print(f"Evaluating Model Loss on Valid Set...")
                            avg_loss = self.ValidOneTime(model)
                            print(f"Saving Model Checkpoint {epoch}-{save_cnt}...")
                            self.SaveModelCkpt(f"{epoch}-{save_cnt}", avg_loss)

                            save_cnt += 1
                        ii += 1
                        # self.SaveTrainerStatus()
            else:
                for _, data in enumerate(self.pretrainloader):
                    print(f"iter: {ii}")
                    if ii < self.start_ii:
                        continue
                    MolEmbeddings, FragViewEmbeddings, FragEmbedsSim = model(data)

                    loss = self.CalculateLoss(MolEmbeddings, FragViewEmbeddings, FragEmbedsSim, data[0].singlebond_num)
                    loss.backward()
                    # update the parameters
                    if (ii+1) % self.opt.args['UpdateRate'] == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    cum_loss += loss.detach()

                    # Print the loss
                    if (ii+1) % self.opt.args['PrintRate'] == 0:
                        print("Loss: ", cum_loss.item() / self.opt.args['PrintRate'])
                        cum_loss = 0.0

                    if (ii+1) % self.opt.args['SaveCkptRate'] == 0:
                        # save ckpt
                        print(f"Evaluating Model Loss on Valid Set...")
                        avg_loss = self.ValidOneTime(model)
                        print(f"Saving Model Checkpoint {epoch}-{save_cnt}...")
                        self.SaveModelCkpt(f"{epoch}-{save_cnt}", avg_loss)

                        save_cnt+=1
                    ii += 1
                    # self.SaveTrainerStatus()
        else:
            raise RuntimeError

        self.lr_sch.step()

    def CalculateLoss(self, MolEmbeddings, FragViewEmbeddings, FragEmbedsSim, singlebond_nums):
        # FragEmbedsSim   [sum(fragpairnum), sum(fragpairnum)], fragpairnum = singlebondnum[i]
        # MolEmbeddings   [BatchSize, FPSize]
        # FragViewEmbeddings   [BatchSize, FPSize]
        # singlebond_nums: number of fragpairs for each mol
        gamma = self.opt.args['L2LossGamma']
        batch_size = MolEmbeddings.size()[0]
        assert FragViewEmbeddings.size()[0] == batch_size

        assert FragEmbedsSim.size()[0] == sum(singlebond_nums)

        # L2 loss of FragEmbedsSim
        if gamma != 0:
            start_idx = 0
            cumulated_sim_num = 0
            all_answers = []
            all_labels = []
            for i in range(batch_size):
                cur_sample_sims = FragEmbedsSim[start_idx:(start_idx+singlebond_nums[i]),start_idx:(start_idx+singlebond_nums[i])]
                # cur_sample_sims: [fragpairnum, fragpairnum]
                # cur_sample_diag_tmp = cur_sample_sims.diag()
                # cur_sample_mask = t.diag(cur_sample_diag_tmp)
                # cur_sample_sims = cur_sample_sims - cur_sample_mask
                # cur_sample_sims: ele on diag are 0, and others are dot similarities between fragpairs.
                # number of cur_sample_sims: fragpairnum * fragpairnum - fragpairnum
                all_answers.append(cur_sample_sims.sum().view(1))
                all_labels.append(1.0)
                cumulated_sim_num += singlebond_nums[i] * singlebond_nums[i]
                start_idx += singlebond_nums[i]

            all_answers = t.cat(all_answers)
            all_labels = t.Tensor(all_labels).to(self.device)
            L2Loss = self.L2Loss(all_answers, all_labels)
            L2Loss = L2Loss / cumulated_sim_num
        else:
            L2Loss = 0

        # NT-Xent Loss of mol-view and frag-view
        CLRLoss = self.NTXentLoss(MolEmbeddings, FragViewEmbeddings)

        # sum loss
        loss = gamma * L2Loss + CLRLoss

        print(f"L2Loss:  {L2Loss},   CLRLoss:  {CLRLoss}")

        return loss

    ################################################################################################################
    def ValidOneTime(self, model):
        model.eval()
        cum_loss = 0.0
        ii = 0
        for data in self.pretrainevalloader:
            MolEmbeddings, FragViewEmbeddings, FragEmbedsSim = model(data)
            loss = self.CalculateLoss(MolEmbeddings, FragViewEmbeddings, FragEmbedsSim, data[0].singlebond_num)
            cum_loss += loss.detach()
            ii += 1
        avg_loss = cum_loss / ii
        print(f"Average loss: {avg_loss}")
        model.train()
        return avg_loss


    def WeightInit(self):
        for param in self.net.parameters():
            self.initer.WeightInit(param)

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
        # os.listdir returns the file names in Addr, only the names, without the Addr path.
        if dir_files:
            dir_files = sorted(dir_files, key=lambda x: os.path.getctime(os.path.join(Addr, x)))
            last_file = dir_files[-1]
        else:
            last_file = ' '
        return last_file

class FinetuneTrainer(object):
    def __init__(self, opt):
        super(FinetuneTrainer, self).__init__()
        self.opt = opt

        t.manual_seed(self.opt.args['TorchSeed'])
        #statussaver = ControllerStatusSaver(self.opt.args, 'Trainer')

        self.device = t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')

        self.pretrained_net, self.net = self.BuildModel()
        self.BuildIniter()
        if self.initer:
            self.WeightInit(self.net)
            if self.opt.args['FromScratch']:
                self.WeightInit(self.pretrained_net)
            # only initialize the weight of downstream predictive head
        self.BuildOptimizer()
        self.StartEpoch = 0

        self.BuildDataset()
        self.BuildCriterion()
        self.evaluator = self.BuildEvaluator()
        self.earlystopcontroller = EarlyStopController(self.opt)

        self.saver = Saver()
        self.controllerstatussaver = ControllerStatusSaver(self.opt.args, 'Trainer', restart=True)


    ########################################################
    def BuildModel(self):
        print(f"Load Pretarined Model from ckpt...")
        pretrained_net = t.load(self.opt.args['PretrainedModelPath'], map_location = 'cpu')['model']
        pretrained_net = pretrained_net.to(self.device)
        net = DNN(
                    self.opt.args['FPSize'],
                    self.opt.args['DNNLayers'],
                    self.opt.args['OutputSize'],
                    self.opt
        ).to(self.device)

        # print(f"PretrainedModel.device:{next(pretrained_net.parameters()).device}")
        return (pretrained_net,net)

    def BuildIniter(self):
        init_type = self.opt.args['WeightIniter']
        if init_type == 'Norm':
            self.initer = NormalInitializer(self.opt)

        elif init_type == 'XavierNorm':
            self.initer = XavierNormalInitializer()
        else:
            self.initer = None

    def BuildEvaluator(self):
        evaluator = PyGFraGATContraFinetuneEvaluator(self.opt)
        return evaluator

    def BuildDataset(self):

        moldatasetcreator = MolDatasetCreator(self.opt)
        sets, self.weights = moldatasetcreator.CreateDatasets()

        if len(self.opt.args['SplitRate']) == 2:
            (Trainset, Validset, Testset) = sets
        elif len(self.opt.args['SplitRate']) == 1:
            (Trainset, Validset) = sets
        else:
            (Trainset) = sets

        self.batchsize = self.opt.args['BatchSize']

        assert len(Trainset) >= self.batchsize

        self.trainloader = tg.loader.DataLoader(Trainset, batch_size = self.batchsize, shuffle = True, num_workers = 8, \
                                              drop_last = True, worker_init_fn = np.random.seed(8), pin_memory = True)
        self.validloader = tg.loader.DataLoader(Validset, batch_size = 64, shuffle = False, num_workers = 0, \
                                              drop_last = False, worker_init_fn = np.random.seed(8), pin_memory = True)
        if len(self.opt.args['SplitRate']) == 2:
            self.testloader = tg.loader.DataLoader(Testset, batch_size = 64, shuffle = False, num_workers = 0, \
                                             drop_last = False, worker_init_fn = np.random.seed(8), pin_memory = True)
        else:
            self.testloader = None


    def BuildOptimizer(self):
        if self.opt.args['FinetunePTM']:
            self.PTMoptimizer = optim.Adam(self.pretrained_net.parameters(), lr=10 ** -self.opt.args['PTMlr'],
                                           weight_decay = 10 ** -self.opt.args['WeightDecay'])
            self.DNNoptimizer = optim.Adam(self.net.parameters(), lr=10 ** -self.opt.args['lr'],
                                           weight_decay = 10**-self.opt.args['WeightDecay'])
            self.optimizer = [self.PTMoptimizer, self.DNNoptimizer]
        else:
            self.optimizer = [optim.Adam(self.net.parameters(), lr = 10 ** -self.opt.args['lr'],
                               weight_decay = 10 ** -self.opt.args['WeightDecay'])]

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


            self.TrainOneEpoch((self.pretrained_net, self.net), self.trainloader, self.validloader, self.testloader, self.optimizer, self.criterion, self.evaluator)
            stop_flag = self.ValidOneTime(epoch)

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



    ################################################################################################################

    def TrainOneEpoch(self, model, trainloader, validloader, testloader, optimizer, criterion, evaluator):
        cum_loss = 0.0     # cum_loss is used to store the entire loss of a print period for printing the average loss.

        self.pretrained_net, self.net = model

        if not self.opt.args['FinetunePTM']:
            # print(f"Set pretrained model to be eval mode...")
            self.pretrained_net.eval()

        for ii, data in enumerate(trainloader):
            # data = data.to(self.device)

            # print(f"data:{data}")

            Label = data[0].y
            Label = Label.t()               # [task, batch]
            Label = Label.to(self.device)
            # data = (None, data, None)
            # print(f"data:{data}")
            # raise RuntimeError
            # if self.opt.args['UseWhichView'] == 'Mol':
            #     data = (None, data[1], None)
            # elif self.opt.args['UseWhichView'] == 'Frag':
            #     data = (data[0],data[1],None)
            data = (data[0].to(self.device),data[1].to(self.device),data[2])
            # print(f"data[0].singlebond_num:{data[0].singlebond_num}")
            # print(f"data[0].x:{data[0].x}")
            # print(f"data[0].edge_attr:{data[0].edge_attr}")
            # print(f"data[0].edge_index:{data[0].edge_index}")

            output = self.pretrained_net(data,finetune=True, usewhichview= self.opt.args['UseWhichView'])
            # print(f"output:{output}")
            if not self.opt.args['FinetunePTM']:
                output = output.detach()

            # print(f"output.requires_grad:{output.requires_grad}")
            output = self.net(output)
            # print(f"Label:{Label}")
            # raise RuntimeError
            loss = self.CalculateLoss(output, Label, criterion)
            loss.backward()

            # update the parameters
            if (ii+1) % self.opt.args['UpdateRate'] == 0:
                for item in optimizer:
                    item.step()
                    item.zero_grad()

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
    def ValidOneTime(self,epoch):
        if self.opt.args['ClassNum'] == 1:
            print('Running on Valid set')
            result = self.evaluator.eval(self.validloader, (self.pretrained_net,self.net), [MAE(), RMSE()])
            if self.testloader:
                print('Running on Test set')
                testresult = self.evaluator.eval(self.testloader, (self.pretrained_net,self.net), [MAE(), RMSE()])

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


            result = self.evaluator.eval(self.validloader, (self.pretrained_net,self.net), [AUC(), ACC()])
            if self.testloader:
                print("running on Test set.")
                testresult = self.evaluator.eval(self.testloader, (self.pretrained_net,self.net), [AUC(), ACC()])

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


    def WeightInit(self,net):
        for param in net.parameters():
            self.initer.WeightInit(param)

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


