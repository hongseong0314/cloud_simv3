from ml_collections import config_dict
def base_config():
    cfg = config_dict.ConfigDict()
    cfg.machines_number = 5
    cfg.jobs_len = 10
    cfg.n_iter = 100
    cfg.n_episode = 12
    cfg.jobs_csv = './independent_job/jobs.csv'

    return cfg

def matrix_config():
    cfg = base_config()
    cfg.model_params = {
                        'embedding_dim': 128,
                        'sqrt_embedding_dim': 128**(1/2),
                        'encoder_layer_num': 3,
                        'qkv_dim': 8,
                        'sqrt_qkv_dim': 16**(1/2),
                        'head_num': 16,
                        'logit_clipping': 10,
                        'ff_hidden_dim': 256,
                        'ms_hidden_dim': 16,
                        'ms_layer1_init': (1/2)**(1/2),
                        'ms_layer2_init': (1/16)**(1/2),
                        'eval_type': 'softmax',
                        'one_hot_seed_cnt': 4,  
                        'nT':4,
                        'nM':2,
                    }
    cfg.optimizer_params = {
                        'optimizer': {
                            'lr': 1e-4,
                            'weight_decay': 1e-6
                        },
                        'scheduler': {
                            'milestones': [101, 151],
                            'gamma': 0.1
                        }
                    }
    return cfg