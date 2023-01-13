from importlib import import_module

def resolve_classpath(p):
    if '.' not in p:
        return import_module(p)

    parts = p.rsplit('.')
    mod = import_module('.'.join(parts[:-1]))
    return getattr(mod, parts[-1])
