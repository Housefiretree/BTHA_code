import torch
from copy import deepcopy

class ModelEMA:
    def __init__(self, model, decay=0.999, device=None):
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                # --- 关键修复 ---
                # 检查设备是否一致，如果不一致，将 model_v 移动到 ema_v 的设备
                if ema_v.device != model_v.device:
                    model_v = model_v.to(device=ema_v.device)
                # ----------------
                
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)