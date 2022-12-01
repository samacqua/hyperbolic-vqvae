import argparse
from parse_config import ConfigParser, _update_config
import train
import os
from utils import read_json
import itertools
from datetime import datetime
import shutil
import torch


torch.set_default_tensor_type(torch.DoubleTensor)


def main(cfgs):

    print("Running", len(cfgs), "experiments.")

    for i, cfg in enumerate(cfgs):

        print(f"Training {cfg.run_id} ({i+1}/{len(cfgs)})")
        print(cfg['name'], end='\n\n')
        train.train(cfg)


def parse_experiment_config(config_path: str):
    """Parses a config file that specifies an experiment to run.

    Can accept a list of config files + edits to make to the files. Runs cartesian product of the edits.
    """

    exp_config = read_json(config_path)
    exp_timestamp = datetime.now().strftime(r'%m%d_%H%M%S') if exp_config['name'] != 'temp' else 'temp'
    default_update = {'name': exp_timestamp, 'trainer;save_dir': "saved/" + exp_config["name"] + "/"}

    # # Remove existing temp directory.
    # if exp_config["name"] == "temp":
    #     shutil.rmtree(default_update['trainer;save_dir'])

    exp_configs = []

    # Read each base config file.
    for cfg_path, cfg_info in exp_config['configs'].items():

        path_base_str = cfg_path.split('/')[-1].split('.')[0]

        # Create a cartesian product of all of the edits.
        edit_lists = []
        single_edit_fields = set()
        for edit_field, edit_vals in cfg_info['edits'].items():

            # Keep track of single-edit hyperparameters for naming sake.
            if not isinstance(edit_vals, list) or len(edit_vals) == 1:
                single_edit_fields.add(edit_field)
                edit_vals = [edit_vals] if not isinstance(edit_vals, list) else edit_vals

            edit_lists.append([(edit_field, val) for val in edit_vals])

        edit_combinations = list(itertools.product(*edit_lists))

        # Apply each combination of edits.
        for edit_combo in edit_combinations:

            # Create string to describe run w/in experiment.
            # Made from name of base config + unique edits.
            edit_str = path_base_str + '_' + '_'.join([str(k.split(';')[-1]) + "=" + str(v) for k, v in edit_combo if k not in single_edit_fields])

            # Parse the base config + make edits.
            update_dict = {**dict(edit_combo), **default_update}
            cfg = ConfigParser(read_json(cfg_path), run_id=edit_str, modification=update_dict)
            exp_configs.append(cfg)

    return exp_configs


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='experiment config file path (default: None)')

    cfgs = parse_experiment_config(args.parse_args().config)
    main(cfgs)
