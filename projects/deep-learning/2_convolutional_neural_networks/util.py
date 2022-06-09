import torch
import numpy as np
from torch import as_strided

def view_as_window(arr_in: torch.Tensor, window_shape: tuple, step: tuple) -> torch.Tensor:
    """Rolling window view of the input n-dimensional array.

        Windows are overlapping views of the input array, with adjacent windows
        shifted by a single row or column (or an index of a higher dimension).
        Parameters
        ----------
        arr_in : torch.Tensor
            N-d input tensor.
        window_shape : integer or tuple of length arr_in.ndim
            Defines the shape of the elementary n-dimensional orthotope
            (better know as hyperrectangle [1]_) of the rolling window view.
            If an integer is given, the shape will be a hypercube of
            sidelength given by its value.
        step : integer or tuple of length arr_in.ndim
            Indicates step size at which extraction shall be performed.
            If integer is given, then the step is uniform in all dimensions.
        Returns
        -------
        arr_out : torch.Tensor
            (rolling) window view of the input tensor.
        Notes
        -----
        One should be very careful with rolling views when it comes to
        memory usage.  Indeed, although a 'view' has the same memory
        footprint as its base array, the actual array that emerges when this
        'view' is used in a computation is generally a (much) larger array
        than the original, especially for 2-dimensional arrays and above.
        For example, let us consider a 3 dimensional array of size (100,
        100, 100) of ``float64``. This array takes about 8*100**3 Bytes for
        storage which is just 8 MB. If one decides to build a rolling view
        on this array with a window of (3, 3, 3) the hypothetical size of
        the rolling view (if one was to reshape the view for example) would
        be 8*(100-3+1)**3*3**3 which is about 203 MB! The scaling becomes
        even worse as the dimension of the input array becomes larger.

        References
        ----------
        .. [1] https://en.wikipedia.org/wiki/Hyperrectangle

        Source
        ------
        This code is inspired by the code for 
        `skimage.util.view_as_windows(arr_in, window_shape, step=1)`. Also, this
        doc-string was copied from the doc-string for the original method.
    """

    if not isinstance(arr_in, torch.Tensor):
        raise TypeError("`arr_in` must be a PyTorch Tensor")

    ndim = arr_in.ndim

    if len(window_shape) != ndim:
        raise ValueError("`window_shape` is incompatible with `arr_in.shape`")

    if len(step) != ndim:
        raise ValueError("`step` is incompatible with `arr_in.shape`")

    arr_shape = torch.Tensor(tuple(arr_in.shape))
    win_shape = torch.Tensor(window_shape)

    if (arr_shape - win_shape < 0).any():
        raise ValueError("`win_shape` is too large")

    if (win_shape - 1 < 0).any():
        raise ValueError("`win_shape` is too small")

    # -- build rolling window view
    win_indices_shape = (torch.div(input=arr_shape - win_shape, 
        other=torch.Tensor(step), rounding_mode="floor") + 1)
    new_shape = tuple(win_indices_shape.int().tolist() + win_shape.int().tolist())

    slices = tuple(slice(None, None, st) for st in step)
    window_strides = arr_in.stride()
    indexing_strides = arr_in[slices].stride()    
    strides = tuple(list(indexing_strides) + list(window_strides))

    arr_out = as_strided(arr_in, size=new_shape, stride=strides)

    return arr_out

def convolved_size(a_shape: tuple, k_shape: tuple, 
        p_shape: tuple, s_shape: tuple) -> tuple:
    '''Compute size of the tensor after convolution.

    a_shape: shape of the input tensor
    k_shape: shape of the kernel tensor. The number of kernels must not be
        included in this size, i.e. shape -> (h_k, w_k, c_k) for 3D tensors
    p_shape: tuple of padding along each of the input tensors' dimensions
    s_shape: tuple of stride along each of the input tensors' dimensions

    out: tuple of size of the output tensor. This shape doesn't include 
        the number of output channels, i.e. shape -> (h_o, w_o, 1) for 3D 
        tensors
    '''
    d = len(a_shape)
    
    if len(k_shape) != d or len(s_shape) != d:
        raise ValueError(
            f"Dimensionality of kernel/stride doesn't match: \
            \n a: {a_shape}\n s: {s_shape}\n k: {k_shape}")
    
    if len(p_shape) != d and len(p_shape) != 2*d:
        raise ValueError(
            f"Dimensionality of padding doesn't match: \
            \n a: {a_shape}\n p: {p_shape}")
    
    padding = 2 * np.array(p_shape) if len(p_shape) == d else \
        np.sum(np.reshape(np.array(p_shape), newshape=(d,2)), 
            axis=1, keepdims=False)
    z = (np.array(a_shape) + padding 
            - np.array(k_shape)) / np.array(s_shape)

    return tuple(np.int32(i) for i in np.floor(z + 1))

def compute_padding(p_template: tuple, a_shape: tuple, 
        k_shape: tuple, s_shape: tuple) -> tuple:
    '''Compute the size of the padding needed for `same-padding`, i.e.
    padding thickness is such that the size of input stays the same even 
    after convolution.

    p_template: the tuple of padding along each of the input tensors' 
        dimensions. If the padding for any dimension is set to -1, it 
        is considered `same-padding`.

        it could also be a string with either 'same' or 'valid' as its value.
        If 'same': a tuple with length same as length of a_shape is 
            created with all values equal to -1.
        If 'valid': a tuple with length same as length of a_shape is 
            created with all values equal to 0.

    a_shape: shape of the input tensor
    k_shape: shape of the kernel tensor. The number of kernels must not be
        included in this size, i.e. shape -> (h_k, w_k, c_k) for 3D tensors
    s_shape: tuple of stride along each of the input tensors' dimensions

    out: a tuple of padding along each of the input tensors' dimensions
    '''
    if isinstance(p_template, tuple):
        pass
    elif isinstance(p_template, str):
        if p_template == 'same':
            p_template = [-1 for i in range(len(a_shape))]
        elif p_template == 'valid':
            p_template = [0 for i in range(len(a_shape))]
        else:
            raise ValueError(f"Invalid padding template: {p_template}") 
    else:
        raise ValueError(f"Invalid padding template: {p_template}")
    
    final_padding = [None] * 2 * len(a_shape)
    for dim in np.arange(start=len(p_template)-1, stop=-1, step=-1):
        pad = p_template[dim]
        if pad == -1:
            pad_float = ((a_shape[dim] - 1) * s_shape[dim] + k_shape[dim] - a_shape[dim]) / 2
            ceil_pad, floor_pad = np.int32(np.ceil(pad_float)), np.int32(np.floor(pad_float))
            a_out_dim_sym = convolved_size(a_shape=(a_shape[dim],), k_shape=(k_shape[dim],),
                p_shape=(ceil_pad,), s_shape=(s_shape[dim],))

            if a_out_dim_sym[0] - a_shape[dim] == 1:
                a_out_dim_asym = convolved_size(a_shape=(a_shape[dim],), k_shape=(k_shape[dim],),
                    p_shape=(ceil_pad, floor_pad), s_shape=(s_shape[dim],))
                if a_out_dim_asym[0] != a_shape[dim]:
                    raise ValueError(
                        f"Neither symmetric nor asymmetric padding are working.\n" 
                        + f"\t- a_shape: {a_shape[dim]}, a_out_sym: {a_out_dim_sym}, a_out_asym: {a_out_dim_asym}")
                
                final_padding[2 * dim] = ceil_pad 
                final_padding[2 * dim + 1] = floor_pad
            elif a_out_dim_sym[0] == a_shape[dim]:
                final_padding[2 * dim] = ceil_pad 
                final_padding[2 * dim + 1] = ceil_pad
            else:
                raise ValueError(
                    f"Difference of a_out_sym and a_shape[dim] not equal to 1.\n" 
                    + f"\t- a_shape: {a_shape[dim]}, a_out_sym: {a_out_dim_sym}")
        else:
            final_padding[2 * dim] = pad 
            final_padding[2 * dim + 1] = pad

    return tuple(final_padding)

def linear(W: torch.Tensor, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    W: (n^{l-1},n^{l})
    b: (n^{l},1)
    A: *(m, n^{l-1}); 
    
    * shape after squeezing out unit-dimensions.

    output: (m, n^{l})
    """
    return torch.transpose(
        torch.matmul(W.T, torch.transpose(torch.squeeze(A), dim0=0, dim1=1)) + b,
        dim0=0, dim1=1)

def logistic(Z: torch.Tensor, L: float=1.0, k: float=1.0, x0: float=0.0, **kwargs) -> torch.Tensor:
    """
    Z: input of any-shape
    L: the curve's maximum (default 1.0)
    k: growth rate (default 1.0)
    x0: the mid-piont (default 0.0)

    output: same shape as the input, and activated element-wise
    """
    return 1 / (1 + torch.exp(-Z))

def relu(Z: torch.Tensor, k: float=0.0, **kwargs) -> torch.Tensor:
    """
    Z: input of any-shape
    k: 0.0 < k <<< 1.0
    
    output: the same shape as input, and recitifed element-wise
    """
    return torch.maximum(Z, torch.tensor(k) * Z)

def softmax(Z: torch.Tensor, dim: int=1, **kwargs) -> torch.Tensor:
    """
    Z: input of any-shape

    output: same shape as input and normalized along the specified "dim" (default is 0)
    """
    sftmx = torch.nn.Softmax(dim=dim)
    return sftmx(Z)

def activation(Zl: torch.Tensor, func_name: str, **kwargs) -> torch.Tensor:
    """
    Zl: shape -> of any shape
    func_name: name of the function, which could be
        'relu': see relu(); see documentation for 
            util.relu() for applicable **kwargs.
        'softmax': see softmax(); see documentation for 
            util.softmax() for applicable **kwargs.
        'sigmoid': see sigmoid(); see documentation for
            util.logistic() for applicable **kwargs.
    """
    if func_name == 'relu':
        return relu(Zl, **kwargs)
    elif func_name == 'softmax':
        return softmax(Zl, **kwargs)
    elif func_name == 'sigmoid':
        return logistic(Zl, **kwargs)
    else:
        raise ValueError(f"Unknown activation-function: {func_name}")

def softmax_cost(Al: torch.Tensor, Y: torch.Tensor, **kwargs) -> float:
    """
    Al: *(m, n^{l})
    Y: *(m, n^{l})

    * shape after squeezing out the unit dimensions.

    output: scalar
    """
    Al = torch.squeeze(Al)
    Y = torch.squeeze(Y)

    assert Al.shape == Y.shape

    m = Al.shape[0]
    return torch.multiply(-Y, torch.log(Al)).sum() / m
