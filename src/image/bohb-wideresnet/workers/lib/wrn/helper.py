import os
import subprocess
import json
import ast
import pickle
#from .ops import get_gene_from_config, Genotype


def load_data(dest_dir):
    #training_loss, validation_loss, training_accuracy, validation_accuracy for
    #the last epoch

    info = {}
    with open(os.path.join(dest_dir,'results.txt'), 'r') as fh:
        data = [ast.literal_eval(json.loads(line)) for line in fh.readlines()]

    with open(os.path.join(dest_dir,'log.txt'), 'r') as fh:
        info['config'] = '\n'.join(fh.readlines())

    info['loss'] = [d['train_loss'] for d in data]
    info['error'] = [d['train_top1'] for d in data]
    info['val_error'] = [d['valid_top1'] for d in data]
    info['test_error'] = [d['test_top1'] for d in data]
    return info


def darts_cifar10(config, budget, config_id, directory, darts_source=''):
    """
    Starts training the sampled architecture
    """
    dest_dir = os.path.join(directory, "_".join(map(str, config_id)))
    ret_dict = {'loss': float('inf'),
                'info': None}

    #genotype = str(get_gene_from_config(config)).replace(' ', '')
    #os.environ['DARTS'] = genotype
    #with open(os.path.join(dest_dir, 'genotype.pkl'), 'wb') as f:
    #    genotype = get_gene_from_config(config)
    #    pickle.dump(genotype, f)

    try:
        bash_strings = ["cd %s; /home/siemsj/miniconda3/envs/random_nas/bin/python train.py"%(darts_source),
                        "--cutout --auxiliary",
                        "--save {} --epochs {}".format(dest_dir, budget),
                        "--edge_normal_0 {}".format(config['edge_normal_0'] if 'edge_normal_0' in config else None),
                        "--edge_normal_1 {}".format(config['edge_normal_1'] if 'edge_normal_1' in config else None),
                        "--edge_normal_2 {}".format(config['edge_normal_2'] if 'edge_normal_2' in config else None),
                        "--edge_normal_3 {}".format(config['edge_normal_3'] if 'edge_normal_3' in config else None),
                        "--edge_normal_4 {}".format(config['edge_normal_4'] if 'edge_normal_4' in config else None),
                        "--edge_normal_5 {}".format(config['edge_normal_5'] if 'edge_normal_5' in config else None),
                        "--edge_normal_6 {}".format(config['edge_normal_6'] if 'edge_normal_6' in config else None),
                        "--edge_normal_7 {}".format(config['edge_normal_7'] if 'edge_normal_7' in config else None),
                        "--edge_normal_8 {}".format(config['edge_normal_8'] if 'edge_normal_8' in config else None),
                        "--edge_normal_9 {}".format(config['edge_normal_9'] if 'edge_normal_9' in config else None),
                        "--edge_normal_10 {}".format(config['edge_normal_10'] if 'edge_normal_10' in config else None),
                        "--edge_normal_11 {}".format(config['edge_normal_11'] if 'edge_normal_11' in config else None),
                        "--edge_normal_12 {}".format(config['edge_normal_12'] if 'edge_normal_12' in config else None),
                        "--edge_normal_13 {}".format(config['edge_normal_13'] if 'edge_normal_13' in config else None),
                        "--edge_reduce_0 {}".format(config['edge_reduce_0'] if 'edge_reduce_0' in config else None),
                        "--edge_reduce_1 {}".format(config['edge_reduce_1'] if 'edge_reduce_1' in config else None),
                        "--edge_reduce_2 {}".format(config['edge_reduce_2'] if 'edge_reduce_2' in config else None),
                        "--edge_reduce_3 {}".format(config['edge_reduce_3'] if 'edge_reduce_3' in config else None),
                        "--edge_reduce_4 {}".format(config['edge_reduce_4'] if 'edge_reduce_4' in config else None),
                        "--edge_reduce_5 {}".format(config['edge_reduce_5'] if 'edge_reduce_5' in config else None),
                        "--edge_reduce_6 {}".format(config['edge_reduce_6'] if 'edge_reduce_6' in config else None),
                        "--edge_reduce_7 {}".format(config['edge_reduce_7'] if 'edge_reduce_7' in config else None),
                        "--edge_reduce_8 {}".format(config['edge_reduce_8'] if 'edge_reduce_8' in config else None),
                        "--edge_reduce_9 {}".format(config['edge_reduce_9'] if 'edge_reduce_9' in config else None),
                        "--edge_reduce_10 {}".format(config['edge_reduce_10'] if 'edge_reduce_10' in config else None),
                        "--edge_reduce_11 {}".format(config['edge_reduce_11'] if 'edge_reduce_11' in config else None),
                        "--edge_reduce_12 {}".format(config['edge_reduce_12'] if 'edge_reduce_12' in config else None),
                        "--edge_reduce_13 {}".format(config['edge_reduce_13'] if 'edge_reduce_13' in config else None),
                        "--inputs_node_normal_3 {}".format(config['inputs_node_normal_3'] if 'inputs_node_normal_3' in config else None),
                        "--inputs_node_normal_4 {}".format(config['inputs_node_normal_4'] if 'inputs_node_normal_4' in config else None),
                        "--inputs_node_normal_5 {}".format(config['inputs_node_normal_5'] if 'inputs_node_normal_5' in config else None),
                        "--inputs_node_reduce_3 {}".format(config['inputs_node_reduce_3'] if 'inputs_node_reduce_3' in config else None),
                        "--inputs_node_reduce_4 {}".format(config['inputs_node_reduce_4'] if 'inputs_node_reduce_4' in config else None),
                        "--inputs_node_reduce_5 {}".format(config['inputs_node_reduce_5'] if 'inputs_node_reduce_5' in config else None),
                       ]

        print(subprocess.check_output(" ".join(bash_strings), shell=True))
        info = load_data(dest_dir)
        ret_dict = {'loss': info['val_error'][-1],
                    'info': info}

    except:
        print("Entering exception!!")
        raise

    return ret_dict

