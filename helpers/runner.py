import argparse
import torch
import multiprocessing

DEFAULT_ARGS = {
    "--name-suffix": {
        "type": str,
        "default": None,
        "help": 'name suffix'
    },
    "--batch-size": {
        "type": int,
        "default": 64,
        "metavar": 'N',
        "help": 'input batch size for training (default: 64)'
    },
    "--epochs": {
        "type": int,
        "default": 14,
        "metavar": 'N',
        "help": 'number of epochs to train (default: 14)'
    },
    "--lr": {
        "type": float,
        "default": 1.0,
        "metavar": 'LR',
        "help": 'learning rate (default: 1.0)'
    },
    "--force-lr": {
        "default": None,
        "help": 'force learning rate'
    },
    "--gamma": {
        "type": float,
        "default": 0.7,
        "metavar": 'M',
        "help": 'Learning rate step gamma (default: 0.7)'
    },
    "--no-cuda": {
        "action": 'store_true',
        "default": False,
        "help": 'disables CUDA training'
    },
    "--seed": {
        "type": int,
        "default": 42,
        "metavar": 'S',
        "help": 'random seed (default: 1)'
    },
    "--log-interval": {
        "type": int,
        "default": 10,
        "metavar": 'N',
        "help": 'how many batches to wait before logging training status'
    },
    "--data": {
        "type": str,
        "default": '../data',
        "metavar": 'STR',
        "help": 'dataset path'
    },
    "--save-model": {
        "action": 'store_true',
        "default": False,
        "help": 'For Saving the current Model'
    },
    "--load-nn": {
        "type": str,
        "metavar": 'STR',
        "default": None,
        "help": 'dataset path',
        "choices": ['last', 'acc', 'loss', None]
    },
    "--skip-train": {
        "type": bool,
        "metavar": 'bool',
        "default": False,
        "help": 'skip train',
    },
    "--skip-test": {
        "type": bool,
        "metavar": 'bool',
        "default": False,
        "help": 'skip train',
    },
    "--infographics": {
        "type": bool,
        "metavar": 'bool',
        "default": False,
        "help": 'skip train',
    },

}

def runner(description, options):
    args = getArgs(description, options)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'batch_size': args.batch_size}

    torch.manual_seed(args.seed)

    if use_cuda:
        kwargs.update({'num_workers': multiprocessing.cpu_count(),
                       'pin_memory': True,
                       'shuffle': True})

    printInfo(args, device)

    return args, kwargs, device


def printInfo(args, device):
    print('-'*42)
    print('Args:')
    for arg in vars(args):
        print(f' {arg}: {getattr(args, arg)}')

    print(f'Device: {device}')
    print('-'*42)


def getArgs(description: str = "Neural Network"):
    parser = argparse.ArgumentParser(description=description)

    for key, val in DEFAULT_ARGS.items():
        parser.add_argument(key, **DEFAULT_ARGS[key])

    return parser.parse_args()
