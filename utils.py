import torch
from importlib import import_module
from typing import List, Union, Dict
from contextlib import contextmanager

def resolve_classpath(p):
    if '.' not in p:
        return import_module(p)

    parts = p.rsplit('.')
    mod = import_module('.'.join(parts[:-1]))
    return getattr(mod, parts[-1])

def make_transform(aug:Union[str, Dict]):

    if isinstance(aug, dict):
        aug_classpath = list(aug.keys())[0]
        aug_init_args = aug[aug_classpath]
        aug = resolve_classpath(aug_classpath)(**aug_init_args)
    else:
        assert isinstance(aug, str)
        aug = resolve_classpath(aug)()
    return aug

def make_augs(augs: List):
    ret = []
    for aug in augs:
        ret += [make_transform(aug)]
    return torch.nn.Sequential(*ret)

@contextmanager
def evaluating(net):
    '''Temporarily switch to evaluation mode.'''
    istrain = net.training
    try:
        net.eval()
        yield net
    finally:
        if istrain:
            net.train()
