import os
import sys
from opts import parser
# BOHB
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB as BOHB
from bohb import ChallengeWorker
from ops.visualize_bohb import visualize
import configuration
# Troch
import torch
import torchvision
# GPU/CPU statistics
import GPUtil
import psutil
# Plotting
import matplotlib as mpl

mpl.use('Agg')  # Server plotting to Agg
GPUtil.showUtilization()  # Show GPUs
# print(psutil.virtual_memory())

def main():
    ############################################################
    parser_args = parser.parse_args()
    print("------------------------------------")
    print("Environment Versions:")
    print("- Python: {}".format(sys.version))
    print("- PyTorch: {}".format(torch.__version__))
    print("- TorchVison: {}".format(torchvision.__version__))
    args_dict = parser_args.__dict__
    print("------------------------------------")
    print(parser_args.arch + " Configurations:")
    for key in args_dict.keys():
        print("- {}: {}".format(key, args_dict[key]))
    print("------------------------------------")
    ############################################################
    run_id = "bohb_run"
    os.getcwd()
    result_directory = os.path.join(parser_args.working_directory,
                                    'bob_res')
    port = 0
    nic_name = 'lo'
    host = hpns.nic_name_to_host(nic_name)
    ns = hpns.NameServer(run_id=run_id,
                         host=host,
                         port=port,
                         working_directory=parser_args.working_directory)
    ns_host, ns_port = ns.start()
    # time.sleep(2) # Wait for nameserver

    workers = []
    ############################################################
    # Worker setup 
    # if training instead of hpo use only one worker
    if parser_args.training: parser_args.bohb_workers = 1
    for i in range(parser_args.bohb_workers):
        w = ChallengeWorker(
            # Nameserver params
            run_id=run_id, host=host, nameserver=ns_host,
            nameserver_port=ns_port, id=i,
            # timeout=120, sleep_interval = 0.5
        )
        w.run(background=True)
        workers.append(w)

    result_logger = hpres.json_result_logger(
        directory=result_directory, overwrite=True
    )
    cs = configuration.get_configspace(model_name=parser_args.arch)
    ############################################################
    # Run bohb if hop, else run training only
    if not parser_args.training:
        bohb = BOHB(
            configspace=cs,
            eta=parser_args.eta,
            run_id=run_id,
            host=host,
            nameserver=ns_host,
            nameserver_port=ns_port,
            result_logger=result_logger,
            min_budget=parser_args.min_budget,
            max_budget=parser_args.max_budget,
        )
        result = bohb.run(n_iterations=parser_args.bohb_iterations)
        bohb.shutdown(shutdown_workers=True)
        ###########################
        # result visualization
        if result is None:
            try:
              result = hpres.logged_results_to_HBS_result(result_directory)
            except: 
              print("No result file found so can't plot any results")
        # Result visualization
        if result is not None:
            visualize(result, result_directory)
    else: 
        ############################################################
        # Training
        config = cs.get_default_configuration()
        workers[0].compute(config=config, 
                           budget=parser_args.epochs,
                           working_directory=parser_args.working_directory)
    ######################################            
    # After training shut down nameserver                               
    ns.shutdown()
    
if __name__ == '__main__':
    main()
