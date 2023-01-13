"""


@author: Guanming Chen (emilien_chen@buaa.edu.cn)
Created on Dec 18, 2022
"""
import os
#Use nohup to run main.py
#run LightGCN
os.system('python main.py --model LightGCN --augment No --loss BPR --comment vanilla_LightGCN')
#run SGL
os.system('python main.py --model SGL --augment ED --loss BPR_Contrast --temp_tau 0.2 --comment vanilla_SGL_ED')
os.system('python main.py --model SGL --augment RW --loss BPR_Contrast --temp_tau 0.2 --comment vanilla_SGL_ED')
#run SimGCL
os.system('python main.py --model SimGCL --augment No --loss BPR_Contrast --temp_tau 0.2 --comment vanilla_SimGCL')
#run BC-loss (only for LightGCN)
os.system('python main.py --model LightGCN --augment No --loss BC --temp_tau 0.1 --comment vanilla_BCloss_LightGCN')
#run Adaptive loss
os.system('python main.py --model LightGCN --augment No --loss Adaptive --temp_tau 0.2 --alpha 0.5 --lambda1 0.1 --lr 0.001 --weight_decay 1e-4 --init_method Normal --comment Adaptive_LightGCN')
os.system('python main.py --model SGL --augment ED --loss Adaptive --temp_tau 0.2 --alpha 0.5 --lambda1 0.1 --lr 0.001 --weight_decay 1e-4 --init_method Normal --comment Adaptive_SGL')
os.system('python main.py --model SimGCL --augment No --loss Adaptive --temp_tau 0.2 --alpha 0.5 --lambda1 0.1 --lr 0.001 --weight_decay 1e-4 --init_method Normal --comment Adaptive_SimGCL')
#run Adaptive augmentation & loss
os.system('python main.py --model GCLRec --augment Adaptive --loss Adaptive --temp_tau 0.2 --alpha 0.5 --lambda1 0.1 --lr 0.001 --weight_decay 1e-4 --init_method Normal --comment Adaptive_loss_augment_GCLRec')
#run Abalation of GCLRec
os.system('python main.py --model GCLRec --augment Adaptive --loss BPR_Contrast --temp_tau 0.2 --alpha 0.5 --lambda1 0.1 --lr 0.001 --weight_decay 1e-4 --init_method Normal --comment GCLRec_Abalation_Adap_Augment_BPR_Conrtrast')
#GCLRec = adaptive loss + adaptive augment


