import logging 
import os, sys, glob
def get_logger(args):
    # Saving path
    log_dir = os.path.join(args.log_dir, args.dataset, args.checkname if args.checkname else '')
    runs = sorted(glob.glob(os.path.join(log_dir, 'experiment_*')))
    #run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
    exist_run_ids = [int(r.split('_')[-1]) for r in runs]
    run_id = sorted(exist_run_ids)[-1] + 1 if runs else 0

    log_experiment_dir = os.path.join(log_dir, 'experiment_{}'.format(str(run_id)))
    if not os.path.exists(log_experiment_dir):
        os.makedirs(log_experiment_dir)
    log_file = os.path.join(log_experiment_dir, 'log.txt')

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG,
                format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(log_file)
            
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    return logging, log_experiment_dir 

