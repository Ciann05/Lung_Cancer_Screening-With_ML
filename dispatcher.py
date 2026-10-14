import argparse
import json
import random
import os, subprocess
from csv import DictWriter
import multiprocessing
import itertools
import sys

def add_main_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--config_path",
        type=str,
        default="grid_search.json",
        help="Location of config file"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of processes to run in parallel"
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Location of experiment logs and results"
    )

    parser.add_argument(
        "--grid_search_results_path",
        default="grid_results.csv",
        help="Where to save grid search results"
    )

    parser.add_argument(
        "--plco_data_path",
        default="lung_prsn.csv",
        help="Location of PLCO csv passed through to main.py",
    )

    return parser

def get_experiment_list(config: dict) -> list[dict]:
    '''
    Parses an experiment config, and creates jobs. For flags that are expected to be a single item, 
    but the config contains a list, this will return one job for each item in the list.
    
    Args:
        config: experiment configuration dictionary from grid_search.json
        
    Example config:
    {
        "learning_rate": [0.001, 0.01],
        "batch_size": [64, 128],
        "regularization_lambda": [0, 0.1]
    }
    
    Returns:
        jobs: a list of dicts, each of which encapsulates one job.
        Example: [
            {"learning_rate": 0.001, "batch_size": 64, "regularization_lambda": 0},
            {"learning_rate": 0.001, "batch_size": 64, "regularization_lambda": 0.1},
            {"learning_rate": 0.001, "batch_size": 128, "regularization_lambda": 0},
            ...
        ]
    '''
    keys = list(config.keys())
    value_lists = []
    for key in keys:
        values = config[key]
        if not isinstance(values, list):
            values = [values]
        value_lists.append(values)

    jobs = [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]
    return jobs

def worker(args: argparse.Namespace, job_queue: multiprocessing.Queue, done_queue: multiprocessing.Queue):
    '''
    Worker thread for each worker. Consumes all jobs and pushes results to done_queue.
    :args - command line args
    :job_queue - queue of available jobs.
    :done_queue - queue where to push results.
    '''
    while True:
        params = job_queue.get()
        if params is None:
            return
        done_queue.put(launch_experiment(args, params))

def launch_experiment(args: argparse.Namespace, experiment_config: dict) -> dict:
    '''
    Launch an experiment and direct logs and results to a unique filepath.
    
    Args:
        args: command line arguments
        experiment_config: flags to use for this model run. Will be fed into main.py
        
    Returns:
        dict: flags for this experiment as well as result metrics
        
    Example return:
    {
        "learning_rate": 0.001,
        "batch_size": 64, 
        "regularization_lambda": 0.1,
        "train_auc": 0.65,
        "val_auc": 0.62
    }
    '''

    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    exp_name = "_".join(f"{k}={v}" for k, v in sorted(experiment_config.items()))
    exp_name = exp_name.replace(os.sep, "-")
    results_path = os.path.join(args.log_dir, f"{exp_name}_results.json")
    log_path = os.path.join(args.log_dir, f"{exp_name}.log")

    cmd = [
        sys.executable,
        "main.py",
        "--plco_data_path",
        args.plco_data_path,
        "--results_path",
        results_path,
    ]
    for key, value in experiment_config.items():
        cmd.extend([f"--{key}", str(value)])

    print(f"Launching: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with open(log_path, "w") as log_file:
        completed = subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
            env=env,
        )

    if completed.returncode != 0:
        raise RuntimeError(
            f"Experiment failed (exit {completed.returncode}). See log: {log_path}"
        )

    metrics = json.load(open(results_path, "r"))
    results = {**experiment_config, **metrics}
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser = add_main_args(parser)
    args = parser.parse_args()
    return args

def main(args: argparse.Namespace) -> list[dict]:
    print(args)
    config = json.load(open(args.config_path, "r"))
    print("Starting grid search with the following config:")
    print(json.dumps(config, indent=2))

    # Expand the config into one experiment per hyperparameter combination
    experiments = get_experiment_list(config)
    random.shuffle(experiments)

    job_queue = multiprocessing.Queue()
    done_queue = multiprocessing.Queue()

    for exper in experiments:
        job_queue.put(exper)

    # One sentinel per worker so each process exits cleanly
    for _ in range(args.num_workers):
        job_queue.put(None)

    print("Launching dispatcher with {} experiments and {} workers".format(len(experiments), args.num_workers))

    workers = []
    for _ in range(args.num_workers):
        process = multiprocessing.Process(target=worker, args=(args, job_queue, done_queue))
        process.start()
        workers.append(process)

    # Accumualte results into a list of dicts
    grid_search_results = []
    for _ in range(len(experiments)):
        grid_search_results.append(done_queue.get())

    for process in workers:
        process.join()

    keys = grid_search_results[0].keys()

    print("Saving results to {}".format(args.grid_search_results_path))

    writer = DictWriter(open(args.grid_search_results_path, 'w'), keys)
    writer.writeheader()
    writer.writerows(grid_search_results)

    print("Done")
    return grid_search_results

if __name__ == '__main__':
    __spec__ = None
    args = parse_args()
    main(args)