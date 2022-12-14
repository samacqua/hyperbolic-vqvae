import argparse
from parse_config import ConfigParser, _update_config
import train
import os
from utils import read_json
import itertools
from datetime import datetime
import shutil
import torch


torch.set_default_tensor_type(torch.FloatTensor)


def main(cfgs):

    print("Running", len(cfgs), "experiments.")

    for i, cfg in enumerate(cfgs):

        print(f"Training {cfg.run_id} ({i+1}/{len(cfgs)})")
        print(cfg['name'], end='\n\n')
        train.train(cfg)


def get_run_resume_path(run_path: str):
    """Gets the most recently saved checkpoint for a run in an experiment if it exists."""
    checkpoint_prefix = "checkpoint-epoch"
    checkpoint_postfix = ".pth"

    # Get latest checkpoint.
    if not os.path.exists(run_path):
        return None
    latest_epoch_path = None
    latest_epoch_num = -1
    for fname in os.listdir(run_path):
        if checkpoint_prefix in fname:
            assert fname.startswith(checkpoint_prefix) and fname.endswith(checkpoint_postfix)
            epoch_num = int(fname[len(checkpoint_prefix):-len(checkpoint_postfix)])
            if epoch_num > latest_epoch_num:
                latest_epoch_num = epoch_num
                latest_epoch_path = os.path.join(run_path, fname)

    return latest_epoch_path


def parse_experiment_config(args):
    """Parses a config file that specifies an experiment to run.

    Can accept a list of config files + edits to make to the files. Runs cartesian product of the edits.
    """

    # Parse args.
    config_path = args.config
    resume_path = args.resume

    # Resume experiment. Run directories made at initialization, so can just iterate through parent directory.
    if resume_path:
        assert config_path is None, "Resuming disregards config."
        cfgs = []
        for run_name in sorted(list(os.listdir(resume_path))):
            run_path = os.path.join(resume_path, run_name)
            resume_cfg = read_json(os.path.join(run_path, "config.json"))
            run_resume_path = get_run_resume_path(run_path)
            cfgs.append(ConfigParser(resume_cfg, run_id=run_name, resume=run_resume_path, exist_ok_override=True))

        return cfgs

    # Set parameters consistent across runs in experiment.
    exp_config = read_json(config_path)
    num_repeats = exp_config.get("repeats", None)
    num_repeats = None if num_repeats == 1 else num_repeats
    exp_timestamp = datetime.now().strftime(r'%m%d_%H%M%S') if exp_config['name'] != 'temp' else 'temp'

    exp_path = "saved/" + exp_config["name"] + "/"
    default_update = {'name': exp_timestamp, 'trainer;save_dir': exp_path}

    # Remove existing temp directory (for doing testing -- don't care about saving results here).
    # if exp_config["name"] == "temp":
    #     try:
    #         shutil.rmtree(default_update['trainer;save_dir'])
    #     except:
    #         pass

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
            edit_str = edit_str[:-1] if edit_str.endswith('_') else edit_str
            # edit_str += '-float'

            for repeat_num in (range(num_repeats) if num_repeats else [None]):
                run_edit_str = edit_str + (f"_{repeat_num}" if repeat_num is not None else "")

                # Parse the base config + make edits.
                update_dict = {**dict(edit_combo), **default_update}
                cfg = ConfigParser(read_json(cfg_path), run_id=run_edit_str, modification=update_dict)
                exp_configs.append(cfg)

    exp_configs.sort(key=lambda x: x.run_id)
    return exp_configs


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='experiment config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to experiment to resume.')

    cfgs = parse_experiment_config(args.parse_args())
    main(cfgs)
