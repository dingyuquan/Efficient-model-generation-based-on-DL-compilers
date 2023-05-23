# -*-coding:UTF-8-*-
from scripts.logger.lemon_logger import Logger
from scripts.tools.mutator_selection_logic import MCMC, Roulette,RandomMutator,RandomMutant
import os
import numpy as np
from scripts.tools import utils
import shutil

import datetime

import warnings

np.random.seed(20200501)
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""



def generate_mutants(model_name,mut_dir,filename,mutate_num,mutate_ops,python_prefix,mutate_ratio=0.3):
    """
    Generate models using mutate operators and store them in the origin model dir
    origin_model_name: "xxxx.h5"
    """
    mutate_ops = ['WS', 'GF', 'NEB', 'NAI', 'NS', 'ARem', 'ARep', 'LA', 'LC', 'LR', 'LS','MLA','ConvReplace'] if mutate_ops is None else mutate_ops
    mutate_op_history = { k:0 for k in mutate_ops}
    mutate_op_invalid_history = {k: 0 for k in mutate_ops}
    mutant_history = []

    origin_model_name=model_name[:-3] + "_origin0.h5"
    origin_save_path = os.path.join(mut_dir,origin_model_name)
    shutil.copy(src=filename,dst=origin_save_path)
    rule_selector = RandomMutator(mutate_ops)
    seed_selector = RandomMutant([origin_model_name])
    mutant_counter = 0

    while mutant_counter < mutate_num:
        picked_seed = utils.ToolUtils.select_mutant(seed_selector)
        selected_op = utils.ToolUtils.select_mutator(rule_selector, last_used_mutator=None)

        new_seed_name = "{}-{}{}.h5".format(picked_seed[:-3],selected_op,mutate_op_history[selected_op])
        if new_seed_name not in seed_selector.mutants.keys():
            new_seed_path = os.path.join(mut_dir, new_seed_name)
            picked_seed_path = os.path.join(mut_dir,picked_seed)
            mutate_st = datetime.datetime.now()
            mutate_status = os.system("{}/lemon_dc/bin/python3 -m  scripts.mutation.model_mutation_generators --model {} "
                                      "--mutate_op {} --save_path {} --mutate_ratio {}".format(python_prefix,picked_seed_path, selected_op,
                                                                             new_seed_path,mutate_ratio))
            mutate_et = datetime.datetime.now()
            mutate_dt = mutate_et - mutate_st
            h, m, s = utils.ToolUtils.get_HH_mm_ss(mutate_dt)
            print("INFO:Mutate Time Used on {} : {}h, {}m, {}s".format(selected_op, h, m, s))

            if mutate_status == 0:

                mutate_op_history[selected_op] += 1
                print("INFO: Mutation progress {}/{}".format(mutant_counter+1,mutate_num))
                mutant_counter += 1
            else:
                mutate_op_invalid_history[selected_op] += 1
                print("ERROR:Exception raised when mutate {} with {}".format(picked_seed,selected_op))
            print("Mutated op used history:")
            print(mutate_op_history)

            print("Invalid mutant generated history:")
            print(mutate_op_invalid_history)



if __name__=="__main__":
    """
    Suppose your initial model is '/home/user/lemon/mylenet.h5'
    model_name = 'mylenet.h5'
    mut_dir = ''/home/user/lemon'
    filename = '/home/user/lemon/mylenet.h5'
    mutate_num: the number of mutants 
    mutate_ops: list of mutators. default is all ops
    python_prefix: path of your python e.g. /home/anaconda/bin/python
    mutate_ratio: ratio of some mutator
    
    This function will generate mutants at mut_dir
    """
    model_name = "densenet121-imagenet_origin.h5"
    mut_dir = "/home/lemon_proj/IST_LEMON/lemon_outputs/dyq/densenet121-imagenet_origin/H5"
    filename = "/home/lemon_proj/IST_LEMON/origin_model/densenet121-imagenet_origin.h5"
    mutate_num = 1
    mutate_ops = ["MLA"]
    python_prefix = "/home/lemon_proj/anaconda3/envs"
    mutate_ratio = 1.0
    generate_mutants(model_name,mut_dir,filename,mutate_num,mutate_ops,python_prefix,mutate_ratio)

#'WS', 'GF', 'NEB', 'NAI', 'NS', 'ARem', 'ARep', 'LA', 'LC', 'LR', 'LS','MLA'