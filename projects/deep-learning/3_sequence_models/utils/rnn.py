import torch

def gradient_clip(z: torch.Tensor, limits: tuple, 
        scheme: str='element-wise', **kwargs) -> torch.Tensor:
    """Clips the gradients within the specified limits using the specified scheme.

    z: tensor of gradients to be clipped
    limits: a tuple of (lower-limit, upper-limit)
    scheme: the strategy for clipping gradients

    output: the output tensor z, with clipped gardients
    """

    assert limits[0] if limits[0] else -float('inf') \
        < limits[1] if limits[1] else float('inf'), \
            f"Invalid limits: {limits}"

    if scheme == 'element-wise':
        return torch.clamp(z, min=limits[0], max=limits[1])
    else:
        raise ValueError(f"Invalid gradient clipping scheme: {scheme}")

class RNNBlock():
    def __init__(self, model: dict):
        self.model = model
    
    def forward(self):
        pass

    def backward(self):
        pass


class LSTM(RNNBlock):
    def __init__(self, model: dict):
        """Initializes a single LSTM RNN block.

        model - a dictionary of model parameters
        """
        super(LSTM, self).__init__(model=model)

        # parameters
        self.p_Wfa = 'p_Wfa'
        self.p_Wfx = 'p_Wfx'
        self.p_bf = 'p_bf'
        
        self.p_Wua = 'p_Wua'
        self.p_Wux = 'p_Wux'
        self.p_bu = 'p_bu'
        
        self.p_Woa = 'p_Woa'
        self.p_Wox = 'p_Wox'
        self.p_bo = 'p_bo'
        
        self.p_Wca = 'p_Wca'
        self.p_Wcx = 'p_Wcx'
        self.p_bc = 'p_bc'
        
        self.p_Wya = 'p_Wya'
        self.p_by = 'p_by'

        # caches
        self.a_t = 'a_t'
        self.a_t_1 = 'a_t_1'
        self.c_t = 'c_t'
        self.c_t_1 = 'c_t_1'
        self.x_t = 'x_t'
        self.yh_t = 'yh_t'
        
        self.dp_Wfa = 'p_Wfa'
        self.dp_Wfx = 'p_Wfx'
        self.dp_bf = 'p_bf'
        
        self.dp_Wua = 'p_Wua'
        self.dp_Wux = 'p_Wux'
        self.dp_bu = 'p_bu'
        
        self.dp_Woa = 'p_Woa'
        self.dp_Wox = 'p_Wox'
        self.dp_bo = 'p_bo'
        
        self.dp_Wca = 'p_Wca'
        self.dp_Wcx = 'p_Wcx'
        self.dp_bc = 'p_bc'
        
        self.dp_Wya = 'p_Wya'
        self.dp_by = 'p_by'

        self.da_t = 'da_t'
        self.da_t_1 = 'da_t_1'
        self.dc_t = 'dc_t'
        self.dc_t_1 = 'dc_t_1'
        self.dyh_t = 'dyh_t'

    def forward(self, a_t_1: torch.Tensor, c_t_1: torch.Tensor, 
            x_t: torch.Tensor, **kwargs) -> tuple:
        """Computes the forward propagation for the LSTM cell.

        a_t_1 -> (m,n_a,1): the activation from the previous cell
        c_t_1 -> (m,n_a,1): the cache from the previous cell
        x_t -> (m,n_x,1): the input for the current cell
        """

        # shape -> (m,n_a,1)
        z_f = torch.matmul(input=self.model[self.p_Wfa].T, other=a_t_1) \
                +  torch.matmul(input=self.model[self.p_Wfx].T, other=x_t) \
                    + self.model[self.p_bf]
        # shape -> (m,n_a,1)
        G_f = torch.sigmoid(input=z_f)
        
        # shape -> (m,n_a,1)
        z_u = torch.matmul(input=self.model[self.p_Wua].T, other=a_t_1) \
                + torch.matmul(input=self.model[self.p_Wux].T, other=x_t) \
                    + self.model[self.p_bu]
        # shape -> (m,n_a,1)
        G_u = torch.sigmoid(input=z_u)

        # shape -> (m,n_a,1)
        z_o = torch.matmul(input=self.model[self.p_Woa].T, other=a_t_1) \
                + torch.matmul(input=self.model[self.p_Wox].T, other=x_t) \
                    + self.model[self.p_bo]
        # shape -> (m,n_a,1)
        G_o = torch.sigmoid(input=z_o)

        # shape -> (m,n_a,1)
        z_c = torch.matmul(input=self.model[self.p_Wca].T, other=a_t_1) \
                + torch.matmul(input=self.model[self.p_Wcx].T, other=x_t) \
                    + self.model[self.p_bc]
        # shape -> (m,n_a,1)
        c_t = torch.multiply(input=G_u, other=torch.tanh(input=z_c)) \
                + torch.multiply(input=G_f, other=c_t_1)
        # shape -> (m,n_a,1)
        a_t = torch.multiply(input=torch.tanh(c_t), other=G_o)

        # shape -> (m,n_y,1)
        z_y = torch.matmul(input=self.model[self.p_Wya].T, other=a_t) \
                    + self.model[self.p_by]
        # shape -> (m,n_y,1)
        yh_t = torch.special.softmax(input=z_y, dim=1)

        return a_t, c_t, yh_t, {
            'z_f': z_f,
            'G_f': G_f,
            'z_o': z_o,
            'G_o': G_o,
            'z_c': z_c,
            'z_y': z_y,
            self.c_t: c_t,
            self.c_t_1: c_t_1,
            self.a_t: a_t,
            self.a_t_1: a_t_1,
            self.x_t: x_t,
            self.yh_t: yh_t
        }

    def backward(self, da_t: torch.Tensor, dy_t: torch.Tensor, 
            forward_cache_t: dict, device: str, **kwargs) -> tuple:
        n_y = self.model['n_y']

        yh_t = forward_cache_t[self.yh_t]
        # shape -> (m,1,n_y)
        dz_y = torch.matmul(input=dy_t, 
            other=torch.multiply(
                input=torch.reshape(torch.eye(n_y, device=device), shape=(1,n_y,n_y)), 
                other=yh_t) \
            - torch.matmul(input=yh_t, other=torch.transpose(yh_t, dim0=1, dim1=2)))

        # shape -> (m,n_y,n_a)
        a_t = forward_cache_t[self.a_t]
        dW_ya = torch.matmul(
            input=torch.transpose(dz_y, dim0=1, dim1=2), 
            other=torch.transpose(a_t, dim0=1, dim1=2))
        
        # shape -> (m,1,n_y)
        db_y = dz_y

        # shape -> (m,1,n_a)
        da_t = da_t + torch.matmul(input=dz_y, other=self.model[self.p_Wya].T)
        # TODO: complete this method

class Vanilla(RNNBlock):
    def __init__(self, model: dict):
        """Initializes a single vanilla RNN block.

        model - a dictionary of model parameters
        """
        super(Vanilla, self).__init__(model=model)
        
        # parameters
        self.p_Waa = 'Waa'
        self.p_Wax = 'Wax'
        self.p_ba = 'ba'
        self.p_Wya = 'Wya'
        self.p_by = 'by'

        # caches
        self.x_t = 'x_t'
        self.a_t_1 = 'a_t_1'
        self.a_t = 'a_t'
        self.yh_t = 'yh_t'

        self.dyh_t = 'dyh_t'
        self.da_t = 'da_t'
        self.dW_ya = 'dW_ya'
        self.db_y = 'db_y'
        self.dW_ax = 'dW_ax'
        self.dW_aa = 'dW_aa'
        self.db_a = 'db_a'
        self.da_t_1 = 'da_t_1'

    def forward(self, a_t_1: torch.Tensor, x_t: torch.Tensor, **kwargs) -> tuple:
        """Performs forward propagation through a single Vanilla RNN block.

        a_t_1 -> (m,n_a,1)
        x_t -> (m,n_x,1)

        output -> (m,n_a,1)
        """
        # shape -> (m,n_a,1)
        z_a = torch.matmul(input=self.model[self.p_Waa].T, other=a_t_1) \
                + torch.matmul(input=self.model[self.p_Wax].T, other=x_t) \
                    + self.model[self.p_ba]
        # shape -> (m,n_a,1)
        a_t = torch.tanh(z_a)
        # shape -> (m,n_y,1)
        z_y = torch.matmul(input=self.model[self.p_Wya].T, other=a_t) \
            + self.model[self.p_by]

        # shape -> (m,n_y,1)
        y_hat = torch.special.softmax(input=z_y, dim=1)

        return y_hat, a_t, {
            'z_y': z_y,
            'z_a': z_a, 
            self.x_t: x_t,
            self.a_t_1: a_t_1,
            self.a_t: a_t, 
            self.yh_t: y_hat}

    def backward(self, da_t: torch.Tensor, dy_t: torch.Tensor, 
            forward_cache_t: dict, device: str, **kwargs) -> tuple:
        """Performs backward propagation through a single Vanilla RNN block.

        da_t -> (m,1,n_a)
        dy_t -> (m,1,n_y)
        forward_cache_t -> a tuple containing (a_t_1, x_t, z_a, a_t, z_y, y_hat), and in this order

        output: da^{t-1}, (dW_ya, db_y, dW_aa, dW_ax, db_a)^{t}
        """
        a_t_1 = forward_cache_t[self.a_t_1]
        x_t = forward_cache_t[self.x_t]
        a_t = forward_cache_t[self.a_t]
        y_hat = forward_cache_t[self.yh_t]

        n_y = self.model['n_y']

        # shape -> (m,1,n_y)
        dz_y = torch.matmul(input=dy_t,
                other=torch.multiply(torch.eye(n=n_y, device=device).reshape(shape=(1,n_y,n_y)), y_hat) \
                 - torch.multiply(input=y_hat, other=torch.transpose(y_hat,dim0=1,dim1=2)))
        # shape -> (m,n_y,n_a)
        dW_ya = torch.matmul(input=torch.transpose(dz_y, dim1=1, dim0=2), 
            other=torch.transpose(a_t, dim0=1, dim1=2))
        # shape -> (m,1,n_y)
        db_y = dz_y
        # shape -> (m,1,n_a)
        da_t = da_t + torch.matmul(input=dz_y, other=self.model[self.p_Wya].T)
        # shape -> (m,1,n_a)
        dz_a = torch.multiply(input=da_t, 
            other=1 - torch.pow(torch.transpose(input=a_t, dim0=1,dim1=2), 2))
        # shape -> (m,n_a,n_a)
        dW_aa = torch.matmul(input=torch.transpose(input=dz_a, dim0=1,dim1=2), 
            other=torch.transpose(input=a_t_1, dim0=1, dim1=2))
        # shape -> (m,n_a,n_x)
        dW_ax = torch.matmul(input=torch.transpose(input=dz_a, dim0=1,dim1=2), 
            other=torch.transpose(input=x_t, dim0=1, dim1=2))
        # shape -> (m,1,n_a)
        db_a = dz_a
        # shape -> (m,1,n_a)
        da_t_1 = torch.matmul(input=dz_a, other=self.model[self.p_Waa].T)

        return da_t_1, {self.da_t: da_t, 
            self.dyh_t: dy_t, 
            self.dW_ya: dW_ya, 
            self.db_y: db_y, 
            self.dW_aa: dW_aa, 
            self.dW_ax: dW_ax, 
            self.db_a: db_a, 
            self.da_t_1: da_t_1}


class GRU(RNNBlock):
    def __init__(self, model: dict):
        super(GRU, self).__init__(model)
    
    def forward():
        pass

    def backward():
        pass
