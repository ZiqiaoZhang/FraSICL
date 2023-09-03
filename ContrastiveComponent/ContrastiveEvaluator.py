from TrainingFramework.Evaluator import *


class PyGFraGATContraFinetuneEvaluator(Evaluator):
    def __init__(self, opt):
        super(PyGFraGATContraFinetuneEvaluator,self).__init__()
        self.opt = opt
        self.device = t.device(f"cuda:{self.opt.args['CUDA_VISIBLE_DEVICES']}" if t.cuda.is_available() else 'cpu')

    def eval(self, evalloader, model, metrics):
        pretrained_net, net = model
        pretrained_net.eval()
        net.eval()
        All_answer = []
        All_label = []
        for i in range(self.opt.args['TaskNum']):
            All_answer.append([])
            All_label.append([])

        for ii, data in enumerate(evalloader):
            # data = data.to(self.device)
            Label = data[0].y
            Label = Label.t()

            data = (data[0].to(self.device),data[1].to(self.device),data[2])

            # data = (None, data, None)
            output = pretrained_net(data, finetune = True, usewhichview = self.opt.args['UseWhichView'])
            output = net(output)


            for i in range(self.opt.args['TaskNum']):
                cur_task_output = output[:,
                                      i * self.opt.args['ClassNum']: (i + 1) * self.opt.args['ClassNum']]
                    # print(cur_task_output.size())   # [batch_size, ClassNum]
                cur_task_label = Label[i]  # [batch_size]
                    # print(cur_task_label.size())

                cur_task_cur_batch_valid_labels = []
                cur_task_cur_batch_valid_answers = []
                for j in range(len(cur_task_label)):
                    l = cur_task_label[j]
                    if l == -1:
                        continue
                    else:
                        cur_task_cur_batch_valid_labels.append(l.item())
                        cur_task_cur_batch_valid_answers.append(cur_task_output[j].tolist())


                for ii, item in enumerate(cur_task_cur_batch_valid_labels):
                    All_label[i].append(item)
                for ii, item in enumerate(cur_task_cur_batch_valid_answers):
                        # print('item size:')
                        # print(len(item))
                    All_answer[i].append(item)

        scores = {}
        All_metrics = []
        for i in range(self.opt.args['TaskNum']):
                # for each task, the All_label and All_answer contains the samples of which labels are not missing
            All_metrics.append([])
            label = All_label[i]
            answer = All_answer[i]

            assert len(label) == len(answer)
            for metric in metrics:
                result = metric.compute(answer, label)
                All_metrics[i].append(result)
                    # if multitask, then print the results of each tasks.
                if self.opt.args['TaskNum'] != 1:
                    print("The value of metric", metric.name, "in task", i, 'is: ', result)
        average = t.Tensor(All_metrics).mean(dim = 0)  # dim 0 is the multitask dim.
            # the lenght of average is metrics num

        for i in range(len(metrics)):
            scores.update({metrics[i].name: average[i].item()})
            print("The average value of metric", metrics[i].name, "is: ", average[i].item())

        pretrained_net.train()
        net.train()
        return scores