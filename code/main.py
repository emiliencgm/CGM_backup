"""


@author: Guanming Chen (emilien_chen@buaa.edu.cn)
Created on Dec 18, 2022
"""
import dataloader
import precalcul
import world
from world import cprint
from world import cprint_rare
import model
import augment
import loss
import procedure
import torch
from tensorboardX import SummaryWriter
from os.path import join
import time
import visual
from pprint import pprint
import utils
from augment import Homophily

world.make_print_to_file()

utils.set_seed(world.config['seed'])

print('==========config==========')
pprint(world.config)
print('==========config==========')

cprint('[DATALOADER--START]')
datasetpath = join(world.DATA_PATH, world.config['dataset'])
dataset = dataloader.dataset(world.config, datasetpath)
cprint('[DATALOADER--END]')

cprint('[PRECALCULATE--START]')
start = time.time()
precal = precalcul.precalculate(world.config, dataset)
end = time.time()
print('precal cost : ',end-start)
cprint('[PRECALCULATE--END]')

models = {'LightGCN':model.LightGCN, 'SGL':model.SGL, 'SimGCL':model.SimGCL, 'GCLRec':model.GCLRec}
Recmodel = models[world.config['model']](world.config, dataset, precal).to(world.device)

homophily = Homophily(Recmodel)

augments = {'No':None, 'ED':augment.ED_Uniform, 'RW':augment.RW_Uniform, 'Adaptive':augment.Adaptive_Neighbor_Augment}
try:
    augmentation = augments[world.config['augment']](world.config, Recmodel, precal, homophily)
except:
    augmentation = None

losss = {'BPR': loss.BPR_loss, 'BPR_Contrast':loss.BPR_Contrast_loss, 'BC':loss.BC_loss, 'Adaptive':loss.Adaptive_softmax_loss}
total_loss = losss[world.config['loss']](world.config, Recmodel, precal, homophily)

w = SummaryWriter(join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + str([(key,value)for key,value in world.log.items()])))

optimizer = torch.optim.Adam(Recmodel.parameters(), lr=world.config['lr'])
train = procedure.Train(total_loss)
test = procedure.Test()

quantify = visual.Quantify(dataset, Recmodel, precal)


try:
    best_result_recall = 0.
    best_result_ndcg = 0.
    stopping_step = 0

    for epoch in range(world.config['epochs']):
        start = time.time()
        if world.config['if_visual'] == 1 and epoch % world.config['visual_epoch'] == 0:
            cprint("[Visualization]")
            if world.config['if_tsne'] == 1:
                quantify.visualize_tsne(epoch)
            if world.config['if_double_label'] == 1:
                quantify.visualize_double_label(epoch)
        
        cprint('[AUGMENT]')
        if world.config['model'] in ['SGL']:
            augmentation.get_augAdjMatrix()

        cprint('[TRAIN]')
        train.train(dataset, Recmodel, augmentation, epoch, optimizer, w)

        if epoch % 1== 0:
            cprint("[TEST]")
            result = test.test(dataset, Recmodel, precal, epoch, w, world.config['if_multicore'])
            if result["recall"] > best_result_recall:
                stopping_step = 0
                advance = (result["recall"] - best_result_recall)
                best_result_recall = result["recall"]
                # print("find a better model")
                cprint_rare("find a better recall", str(best_result_recall), extra='++'+str(advance))                
                #torch.save(Recmodel.state_dict(), weight_file)
            else:
                stopping_step += 1
                if stopping_step >= world.config['early_stop_steps']:
                    print(f"early stop triggerd at epoch {epoch}, best recall: {best_result_recall}")
                    break
        during = time.time() - start
        print(f"time cost of epoch {epoch}: ", during)
finally:
    w.close()
