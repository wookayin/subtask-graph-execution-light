import numpy as np


def _sample_int_layer_wise(nbatch, high, low):
    assert len(high.shape) == 1 and len(low.shape) ==1
    ndim = len(high)
    out_list = []
    for d in range(ndim):
        out_list.append( np.random.randint(low[d], high[d]+1, (nbatch, 1) ) )
    return np.concatenate(out_list, axis=1)


def _sample_layer_wise(nbatch, high, low):
    assert len(high.shape) == 1 and len(low.shape) ==1
    nsample = len(high)
    base = np.random.rand( nbatch, nsample )
    return base * (high - low) + low


def _transform(input_tensor, mapping):
    if input_tensor.dim()==1:
        input_tensor = input_tensor.unsqueeze(-1)
    return torch.gather(mapping, 1, input_tensor)


def _to_multi_hot(index_tensor, max_dim): # number-to-onehot or numbers-to-multihot
    if type(index_tensor)==np.ndarray:
        index_tensor = torch.from_numpy(index_tensor)
    if len(index_tensor.shape)==1:
        out = (index_tensor.unsqueeze(1) == torch.arange(max_dim).reshape(1, max_dim))
    else:
        out = (index_tensor == torch.arange(max_dim).reshape(1, max_dim))
    return out


def batch_bin_encode_64(bin_tensor):
  # bin_tensor: Nbatch x dim
  assert isinstance(bin_tensor, np.ndarray)
  #assert bin_tensor.dtype == np.bool

  # TODO: This is buggy because it cannot handle ternary (1, 0, -1) inputs
  return bin_tensor.dot(
      (1 << np.arange(bin_tensor.shape[-1]))
  ).astype(np.int64)

