import importlib.resources as resources
import cupy as cp

def load_cuda_source(name: str) -> str:
    """
    Load a .cu file from the package 'tropoi.spatial.transforms.cuda'.

    Example:
        code = load_cuda_source("legendre")
    """
    package = "tropoi.spatial.transforms.cuda"
    with resources.files(package).joinpath(f"{name}.cu").open("r") as f:
        return f.read()

def raw_module_from_cuda(name: str, **kwargs) -> cp.RawModule:
    """
    Convenience wrapper: load <name>.cu and create a RawModule.
    """
    code = load_cuda_source(name)
    return cp.RawModule(code=code, **kwargs)
