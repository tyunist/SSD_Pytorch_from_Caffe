import os
import shutil
import torch
from collections import OrderedDict
import glob
import yaml

class Saver(object):

    def __init__(self, args):
        self.args = args
        self.directory = os.path.join(args.checkpoint_dir, args.dataset, args.checkname)
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        #run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0
        exist_run_ids = sorted([int(r.split('_')[-1]) for r in self.runs])
        run_id = exist_run_ids[-1] + 1 if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, is_best, filename='checkpoint'):
        """Saves checkpoint to disk"""
        epoch = state['epoch']
        if is_best:
            filename = 'best_checkpoint'
            filename = os.path.join(self.experiment_dir, filename + '.pth.tar')
        else:
            filename = filename 
            filename = os.path.join(self.experiment_dir, filename + '_%d.pth.tar'%epoch)

        torch.save(state, filename)
        if is_best:
            best_RMSE = state['best_RMSE']
            with open(os.path.join(self.experiment_dir, 'best_RMSE.txt'), 'w') as f:
                f.write(str(best_RMSE))
            if self.runs:
                previous_RMSE = [1e6]
                for run in self.runs:
                    run_id = run.split('_')[-1]
                    path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_RMSE.txt')
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            RMSE = float(f.readline())
                            previous_RMSE.append(RMSE)
                    else:
                        continue
                min_RMSE = min(previous_RMSE)
                if best_RMSE <= min_RMSE:
                    shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
            else:
                shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
        self.ckpt_path = filename 

    def save_experiment_config(self, p=None):
        ''' Dump configuration parameters to a file
        p: an OrderedDict()'''
        logfile = os.path.join(self.experiment_dir, 'parameters.yml')
        with  open(logfile, 'w') as f:
            yaml.dump(p, f, default_flow_style=False)
        
    
