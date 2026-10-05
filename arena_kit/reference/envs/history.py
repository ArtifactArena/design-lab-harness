"""
Environment state management for action history tracking.
Used by match environments to maintain bot action sequences.
"""
from collections import deque
from typing import Optional
import numpy as np

DEFAULT_HISTORY_LEN = 10


class ActionHistory:
    """Efficient storage and retrieval of recent bot actions during matches."""
    
    def __init__(self, action_dim: int, maxlen: int = DEFAULT_HISTORY_LEN):
        self.action_dim = action_dim
        self.maxlen = maxlen
        self.buf = deque(maxlen=maxlen)

    def append(self, action):
        """Add action to history. Supports list/numpy/torch.Tensor inputs."""
        # supports list/np/torch.Tensor
        if "torch" in globals() and hasattr(action, "detach"):
            action = action.detach().cpu().float().view(-1).numpy()
        else:
            action = np.asarray(action, dtype=np.float32).reshape(-1)

        if action.size != self.action_dim:
            raise ValueError(f"expected {self.action_dim}, got {action.size}")
        self.buf.append(action.reshape(self.action_dim))

    def as_array(self, pad: bool = True) -> np.ndarray:
        """Convert to numpy array format for bot observations."""
        if not self.buf:
            return np.zeros((0, self.action_dim), dtype=np.float32)
        arr = np.stack(self.buf)  # shape: (k, action_dim)
        if pad and arr.shape[0] < self.maxlen:
            out = np.zeros((self.maxlen, self.action_dim), dtype=np.float32)
            out[-arr.shape[0]:] = arr
            return out
        return arr

    def last_k(self, k: int) -> np.ndarray:
        """Get the last k actions."""
        k = min(k, len(self.buf))
        return np.stack(list(self.buf)[-k:]) if k else np.zeros((0, self.action_dim), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.buf)
    
    def __getitem__(self, idx):
        return self.buf[idx]
    
    def __setitem__(self, idx, value):
        self.buf[idx] = value
    
    def __delitem__(self, idx):
        del self.buf[idx]
    
    def __iter__(self):
        return iter(self.buf)
    
    def __next__(self):
        return next(self.buf)
    
    def spaces(self):
        """Return gymnasium space for this action history."""
        from gymnasium import spaces
        return spaces.Box(low=-1.0, high=1.0, shape=(self.maxlen, self.action_dim), dtype=np.float32)