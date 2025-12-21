import torch
from py.dds import *
import ctypes
import numpy as np

def dds_from_tensor(tensor):
    B, C, H, W = tensor.shape
    assert B == 1, "Only single image tensors are supported"

    if C == 3:
        tensor = torch.cat([tensor, torch.ones((1, 1, H, W), device=tensor.device)], dim=1)
        C = 4
    
    if C == 2:
        tensor = torch.cat([tensor,
                            torch.zeros((1, 1, H, W), device=tensor.device),
                            torch.ones((1, 1, H, W), device=tensor.device)], dim=1)
        C = 4

    assert C == 4, "Only 4 channel tensors are supported"
    tensor = tensor.squeeze(0)  # Remove batch dimension
    tensor = tensor.permute(1, 2, 0).contiguous()

    dds                         = DDSTexture()
    dds.header.width            = W
    dds.header.height           = H
    dds.header.mip_map_count    = 1
    dds.dx10_header.dxgi_format = DXGI_FORMAT.R16G16B16A16_FLOAT.value
    dds.dx10_header.resource_dimension = D3D10_RESOURCE_DIMENSION.TEXTURE2D.value
    dds.dx10_header.array_size = 1
    dds.dx10_header.misc_flag = 0
    dds.dx10_header.misc_flag2 = 0

    cpu_buffer  = tensor.detach().cpu().numpy().astype(np.float16)
    buffer      = np.zeros(cpu_buffer.nbytes, dtype=np.uint8)
    ctypes.memmove(buffer.ctypes.data, cpu_buffer.ctypes.data, cpu_buffer.nbytes)
    dds.buf_ref  = BufferWrapper(buffer, 0, buffer.nbytes)
    return dds

import signal
import logging

class DelayedKeyboardInterrupt:

    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)
                
    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        logging.debug('SIGINT received. Delaying KeyboardInterrupt.')
    
    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)


def erf(x):
    # return torch.sign(x) * torch.sqrt(1 - torch.exp(-x**2 * (4 / torch.pi + 0.147 * x**2) / (1 + 0.147 * x**2)))
    return torch.sign(x) * torch.sqrt(1 - torch.exp(-x**2 * 1.24))
