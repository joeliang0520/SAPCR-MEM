"""Set deterministic defaults inside the DA3 subprocess when requested.

Python imports this module at startup when its directory is on PYTHONPATH.
Da3Runner sets that path and VWS_DETERMINISM for deterministic runs only.
"""

import os

if os.environ.get("VWS_DETERMINISM") == "1":
    import random

    import numpy
    import torch

    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
