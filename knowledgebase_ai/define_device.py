import torch

def define_device():
    """Define the device to be used by PyTorch"""

    #Get the PyTorch version
    torch_version = torch.__version__

    # Print the PyTorch version
    print(f"PyTorch version: {torch_version}", end=" -- ")

    #check if cuda is available
    if torch.cuda.is_available():
        defined_device = torch.device("cuda")
    else:
        defined_device = torch.device("cpu")

    print(f"using {defined_device}")

    return defined_device