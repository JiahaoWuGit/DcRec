import os
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
import utils
import multiprocessing
import time
from os.path import join
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# torch.cuda.synchronize()
def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")

def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def main(config = {}, path_config ={}, test_epochs=10, dataset_select =None, tfBoard_enable=None, device = None):
    # =============Path============
    set_seed(config['SEED'])
    if not os.path.exists(path_config['FILE_PATH']):
        os.makedirs(path_config['FILE_PATH'], exist_ok=True)
    weight_file = join(path_config['FILE_PATH'], "DCCLoss.pth.tar")
    sys.path.append(join(path_config['CODE_PATH'], 'sources'))
    # =============Tensorboard============
    if tfBoard_enable:
        tf_writer: SummaryWriter = SummaryWriter(
            join(BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-"))
        )
    else:
        tf_writer = None
        print("=======not enable tensorflowboard")


    # ==========Loading data==========
    import dataloader
    start = time.time()
    if dataset_select == 'lastfm' or dataset_select == 'douban' or dataset_select == 'filmtrust':
        dataset = dataloader.Loader(dataset_select, device, config['ratioLowPreserve'], config['prejudice'])
    else:
        dataset = dataloader.BigLoader(dataset_select, device, config['ratioLowPreserve'], config['prejudice'])
    end = time.time()
    print('Loading data time(seconds))', int(int((end-start)*1000)/1000),'.',int((end-start)*1000%1000))

    # =============Model==============
    import model
    start = time.time()
    encoder = model.Model(config, dataset).to(device)#tempConf.device
    print('Model initialization time(seconds):', int(int((end-start)*1000)/1000),'.',int((end-start)*1000%1000))
    # =============Loss============
    dcc = utils.DCCLoss(encoder, config)
    # =============LossCalculation============
    test_result = []
    train_infos = []
    try:
        TRAIN_epochs = config['TRAIN_epochs']
        for epoch in range(TRAIN_epochs):
            #start = time.time()
            if epoch % test_epochs == 0:
                cprint("[TEST]")
                sub_result = Procedure.Test(dataset, encoder, epoch, tf_writer, config['multicore'])
                if epoch<config['PRETRAIN_epochs'] and config['pretrain']:
                    sub_result = 'Pretraining:'+str(sub_result)
                test_result.append(sub_result)

            training_information = Procedure.DCC_train_original(dataset, encoder, dcc, epoch,
                                                                tf_writer=tf_writer)  
            train_info = f'EPOCH[{epoch + 1}/{TRAIN_epochs}] {training_information}'
            print(train_info)
            train_infos.append(train_info+'\n')
            torch.save(encoder.state_dict(), weight_file)
    finally:
        if tfBoard_enable:
            tf_writer.close()
    return config, test_result, train_infos

def testResultStore(dataset, root_path, test_epochs, config, result, time_total, train_info):
    dataName = str(dataset)
    print_result = []
    i = 0
    for r in result:
        print_result.append(str(i)+'th Epoch: '+str(r)+'\n')
        i = i+test_epochs
    if config['ssl_enable']:
        if config['social_enable']:
            model = 'CycleContra'
        else:
            if config['sglOrsgls'] == 'sgl':
                if config['lOrls'] == 'lightGCN':
                    model = 'ssl_LightGCN'
                else:
                    model = 'SGL-plus-social'
            else:
                model = 'social_SGL'
    else:
        if config['lOrls'] == 'lightGCN':
            model = 'LightGCN'
        else:
            model = 'LightGCN_Social'
    localtime = str(time.asctime(time.localtime(time.time())))
    root_path = join(root_path)
    fileName =  dataName+ '_' + model + '_' + localtime + '.txt'
    path = join(root_path, fileName)
    f_r = ['Testing model:'+model, '\nDataset:'+dataName, '\nTesting time:'+localtime, '\nConfiguration:{\n']
    for key in config.keys():
        f_r.append('\t\t'+str(key)+':\t'+str(config[key])+'\n')
    f_r.append('\nResult:\n')
    f_r = f_r + print_result

    f_r.append('Time spent for training and testing(Seconds):'+str(int(time_total)/3600)+'hours,'+str((int(time_total)%3600)/60)+'minutes,'+str(int(time_total)%60)+'seconds')
    f_r.append('\n\nTraining loss:\n\n')
    for train_loss in train_info:
        f_r.append(train_loss)
    with open(path, 'w') as f:
        f.writelines(f_r)


if __name__ == '__main__':
    STORE_ROOT_PATH = '/home/mason/workstation/new_model/result/fast_ciao/remove-semantic-transformation'
    ROOT_PATH = '/home/mason/workstation/new_model/cycleContra'
    CODE_PATH = join(ROOT_PATH, 'code')
    DATA_PATH = join(ROOT_PATH, 'data')
    BOARD_PATH = join(CODE_PATH, 'runs')
    FILE_PATH = join(CODE_PATH, 'checkpoints')
    RESULT_PATH = join(ROOT_PATH, 'testResult')

    iter_tuning = False
    dataset_select = 'dianping' 
    dataset_iter = False
    datasets = ['lastfm', 'filmtrust', 'douban', 'yelp', 'ciao', 'epinions']
    path_config = {
        'ROOT_PATH':ROOT_PATH,
        'CODE_PATH':CODE_PATH,
        'DATA_PATH':DATA_PATH,
        'BOARD_PATH':BOARD_PATH,
        'FILE_PATH':FILE_PATH,
        'RESULT_PATH':RESULT_PATH
    }

    tfBoard_enable = True
    test_epochs = 100
    x_decay = 0.001
    config = {
        # ++++++++++Model Config++++++++++++++
        'ratioLowPreserve': 0.6,
        'prejudice': False,
        'bpr_batch_size': 2048,
        'latent_dim_rec': 128,
        'model_n_layers': 4,
        'social_n_layers': 4,
        'dropout': 0,
        'keep_prob': 0.6,  
        'A_n_fold': 100,  
        'test_u_batch_size': 100,  
        'multicore': 0,
        'lr': 1e-3,  # the learning rate: try 0.001
        'normalization_decay': 1e-3, 
        'cycle_decay': x_decay,
        'interOrIntra': 'interAintra',
        'social_ssl_decay': x_decay,
        'preference_ssl_decay': 0.02,
        'A_split': False,
        'bigdata': False,
        # Preference domain ssl
        'lOrls': 'lightGCN',
        'sglOrsgls': 'sgl',
        'ssl_enable': True,  # contrastive learning or not============
        'aug_type1': 'edge_drop',  # edge_add, edge_drop, node_drop, random_walk, None
        'edge_drop_rate1': 0.2,  
        'node_drop_rate1': 0,  
        # For view2
        'aug_type2': 'node_drop',  # edge_add, edge_drop, node_drop, random_walk, None
        'edge_drop_rate2': 0.3,  
        'node_drop_rate2': 0.,
        # social part
        'social_enable': True,  # ============
        'cycleMLP': False,
        'contrastMLP': False,
        'social_contrastMLP': True,
        'aug_social1': 'edge_add',
        'ed_social1': 0.9,  # 0.1,0.2,0.3,0.4
        'nd_social1': 0,  # 0.1,0.2,0.3,0.4
        'aug_social2': 'edge_drop',  #
        'ed_social2': 0.9,  # 0.1,0.2,0.3,0.4
        'nd_social2': 0.,  # 0.1,0.2,0.3,0.4
        'cycle_hidden_dim': 64,  # 64,
        'dropout_mlp': 0.2,
        'tau': 0.5,  # {0.1, 0.2, 0.5, 1.0},
        # training configs
        'TOPKS': [5, 10, 20],  # [5,10,20]
        'pretrain': False,  # whether we use pretrained weight or not
        'PRETRAIN_epochs': 100,
        'TRAIN_epochs': 1001,
        'DEVICE': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'CORES': multiprocessing.cpu_count(),
        'SEED': 2020
    }

    if not dataset_iter:

        start = time.time()
        if iter_tuning:
            tuningPara = tuningParameters()
        else:
            config, test_result, train_info = main(config, path_config, test_epochs, dataset_select, tfBoard_enable, config['DEVICE'])# return a list of sentence

        end = time.time()
        testResultStore(dataset_select, STORE_ROOT_PATH, test_epochs, config, test_result, (end-start), train_info)
    else:
        for dataset in datasets:
            if iter_tuning:
                tuningPara = tuningParameters()
            else:
                config, test_result, train_inform = main(config, path_config, test_epochs, dataset, tfBoard_enable,
                                           config['DEVICE'])  
            testResultStore(dataset, path_config['RESULT_PATH'], test_epochs, config, test_result)
