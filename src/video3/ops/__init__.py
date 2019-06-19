from .bninception.pytorch_load import BNInception, InceptionV3
from .C3DRes18.pytorch_load import C3DRes18
from .ECO.pytorch_load import ECO
from .ECOfull.pytorch_load import ECOfull

# Stupid code to prevent autoformat
try:
    assert BNInception
    assert InceptionV3
    assert C3DRes18
    assert ECO
    assert ECOfull
except Exception:
    pass
