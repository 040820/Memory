import math
import numpy as np
import torch

def to_jsonable(value):
 if isinstance(value,torch.Tensor):return to_jsonable(value.detach().cpu().tolist())
 if isinstance(value,np.ndarray):return to_jsonable(value.tolist())
 if isinstance(value,np.generic):return to_jsonable(value.item())
 if isinstance(value,dict):return {str(k):to_jsonable(v) for k,v in value.items()}
 if isinstance(value,(list,tuple)):return [to_jsonable(v) for v in value]
 if isinstance(value,float) and not math.isfinite(value):raise ValueError('nonfinite value in result')
 if value is None or isinstance(value,(str,int,float,bool)):return value
 raise TypeError('unsupported result type: '+type(value).__name__)
