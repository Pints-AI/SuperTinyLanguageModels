""" General trainer utils """

import importlib
import inspect
import os, re
import pkgutil
import numpy as np
from prettytable import PrettyTable

import torch 
import torch.distributed as dist

def set_seed(seed):
    """Setup the trainer"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)


def create_folder_structure(path_config):
    """
    Create all necessary folders for training.
    """
    if not os.path.exists(path_config["data_dir"]):
        os.makedirs(path_config["data_dir"])

    if not os.path.exists(path_config["checkpoint_dir"]):
        os.makedirs(path_config["checkpoint_dir"])



def get_classes_from_module(module_name):
    """
    Get a list of classes defined in a module or package.

    Args:
        module_name (str): The name of the module or package.

    Returns:
        list: A list of classes defined in the module or package.
    """
    module = importlib.import_module(module_name)
    classes = []

    for _, obj in inspect.getmembers(module, inspect.isclass):
        if inspect.getmodule(obj) == module:
            classes.append(obj)

    return classes


def get_classes_from_package(package_name):
    """
    Get a list of classes defined in a package and its subpackages.

    Args:
        package_name (str): The name of the package.

    Returns:
        list: A list of classes defined in the package and its subpackages.
    """
    package = importlib.import_module(package_name)
    classes = get_classes_from_module(package_name)

    for _, module_name, _ in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        classes.extend(get_classes_from_module(module_name))

    return classes


def register_backward_hooks(tensor, module_name):
    """Registers hooks to profile the backward pass of a tensor."""
    if isinstance(tensor, torch.Tensor) and tensor.requires_grad:

        def backward_hook(grad):
            with torch.autograd.profiler.record_function(f"{module_name}.backward"):
                return grad

        tensor.register_hook(backward_hook)





def is_dist():
    """
    Check if the current process is distributed.
    """
    return dist.is_initialized()

def aggregate_value(value, device = torch.device("cuda")): 
    """
    Since using DDP, calculation of metrics happen across all GPUs. 
    This function aggregate the loss across all GPUs. 
    """
    if not is_dist():
        return value
    all_loss = torch.tensor([value], device=device)
    dist.all_reduce(all_loss, op=dist.ReduceOp.SUM)
    return all_loss.item() / dist.get_world_size()
    # return value

def init_print_override():
    '''
    Overriding the print function is useful when running DDP. 
    This way, only rank 0 prints to the console.
    '''
    import builtins as __builtin__
    
    original_print = __builtin__.print

    def print(*args, **kwargs):
        if os.getenv('GLOBAL_RANK') == '0':
            original_print(*args, **kwargs)

    __builtin__.print = print

    return original_print

def restore_print_override(original_print):
    '''
    Restore the original print function.
    '''
    import builtins as __builtin__
    __builtin__.print = original_print




def print_evaluation_results(iter_num, eval_results):
    """
    This function processes and visualizes the evaluation results.
    The input format is a dictionary where each key has a logging path/metric_name format.
    Keys without the '/' should be ignored. 
    The function prints tables for each unique logging path.
    """
    
    # Keys to be ignored (e.g., 'token_num', 'iter')
    ignore_keys = set(["token_num", "iter"])
    
    # Filter out keys that don't have a "/"
    valid_keys = {k: v for k, v in eval_results.items() if "/" in k and k.split("/")[0] not in ignore_keys}
    
    # Identify unique logging paths (the part before "/")
    logging_paths = set([k.split("/")[0] for k in valid_keys])
    
    # Dictionary to store tables for each logging path
    tables = {}

    # Loop through each logging path and generate a table
    for table_name in logging_paths:
        # Collect columns for the table (the part after "/")
        columns = sorted(set([k.split("/")[1] for k in valid_keys if table_name == k.split("/")[0]]))
        
        # Initialize a table with the logging path as the category and columns for the metrics
        table = PrettyTable(["Evaluation"] + columns)
        
        # Collect values for the current logging path
        row_values = {col: "" for col in columns}
        for k, v in valid_keys.items():
            logging_path, col_name = k.split("/")
            if logging_path == table_name:
                row_values[col_name] = v
        
        # Add the row to the table
        table.add_row([table_name] + [row_values[col] for col in columns])
        tables[table_name] = table

    # Print all tables
    for table_name, table in tables.items():
        print(f"\nResults for {table_name}:")
        print(table)
