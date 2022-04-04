import numpy

print("hello world")

from tkinter import Y
import torch
from torch import nn 
import numpy as np

class InputMapping(nn.Module):
    """Fourier features mapping
    contains W layer 
    implements wsinbx"""

    #should B be an input parameter in InputMapping?
    def __init__(self, d_in, n_freq, hidden_features, sigma=2, tdiv=2,  Tperiod=None): 
        # removed parameter: incrementalMask=True, added hiddenfeatures
        super(InputMapping, self).__init__()
        Bmat = torch.randn(n_freq, d_in) * np.pi* sigma/np.sqrt(d_in)  # gaussian
        B_first_guess = 

        # time frequencies are a quarter of spacial frequencies.
        # Bmat[:, d_in-1] /= tdiv
        Bmat[:, 0] /= tdiv

        self.Tperiod = Tperiod
        if Tperiod is not None:
            # Tcycles = (Bmat[:, d_in-1]*Tperiod/(2*np.pi)).round()
            # K = Tcycles*(2*np.pi)/Tperiod
            # Bmat[:, d_in-1] = K
            Tcycles = (Bmat[:, 0]*Tperiod/(2*np.pi)).round()
            K = Tcycles*(2*np.pi)/Tperiod
            Bmat[:, 0] = K
        
        Bnorms = torch.norm(Bmat, p=2, dim=1)
        sortedBnorms, sortIndices = torch.sort(Bnorms)
        Bmat = Bmat[sortIndices, :]

        self.d_in = d_in
        self.n_freq = n_freq
        self.d_out = n_freq * 2 + d_in if Tperiod is None else n_freq * 2 + d_in - 1

        self.B = nn.Linear(d_in, self.d_out, bias=False)
        self.W = nn.Linear(self.d_out, hidden_features)
        with torch.no_grad():
            self.B.weight = nn.Parameter(Bmat.to(device), requires_grad=False)
            self.mask = nn.Parameter(torch.zeros(
                1, n_freq), requires_grad=False)

        # self.incrementalMask = incrementalMask
        # if not incrementalMask:
            # self.mask = nn.Parameter(torch.ones(
            #     1, n_freq), requires_grad=False)

    # def step(self, progressPercent):
    #     if self.incrementalMask:
    #         float_filled = (progressPercent*self.n_freq)/.7
    #         int_filled = int(float_filled // 1)
    #         remainder = float_filled % 1

    #         if int_filled >= self.n_freq:
    #             self.mask[0, :] = 1
    #         else:
    #             self.mask[0, 0:int_filled] = 1
    #             # self.mask[0, int_filled] = remainder


    def forward(self, xi, hidden_features):
        # pdb.set_trace()
        dim = xi.shape[1]-1
        y = self.B(xi)
        if self.Tperiod is None:
            net = torch.cat([torch.sin(y)*self.mask, torch.cos(y)*self.mask, xi], dim=-1)
        else:
            net = torch.cat([torch.sin(y)*self.mask, torch.cos(y)*self.mask, xi[:,1:dim+1]], dim=-1)
        net.append(nn.Linear(self.d_out), hidden_features) #W layer

        X = np.matmul(net, self.W)
        return net

class velocMLP(nn.Module): 
    """
    implements f_th()
    """
    def __init__(self, in_features=3, hidden_features=512, hidden_layers=2,
                 out_features=2, sigmac=3, n_freq=70, tdiv=1,
                  Tperiod=None): #removed param incrementalMask=True,
        super(velocMLP, self).__init__()

        net = []
        imap = InputMapping(in_features, n_freq, sigma=sigmac,
                            tdiv=tdiv, Tperiod=Tperiod)
        self.imap = imap
        net.append(imap)

        #deleted linear layer W, put in InputMapping
        # net.append(nn.Linear(imap.d_out, hidden_features)) 

        for i in range(hidden_layers):
            net.append(nn.Tanh())
            net.append(nn.Linear(hidden_features, hidden_features))
        net.append(nn.Softplus())
        net.append(nn.Linear(hidden_features, out_features))
        net = nn.Sequential(*net)
        self.f = net

    def get_z_dot(self, W, z):
        """
        gets df/dw
        z_dot is parameterized by a NN: z_dot = NN(t, z(t))
        z_dot is the velocity, and is a function of x, y, t (not a constant value)
        df/dw is a function of (x, y) coordinate, W
        replace t with W to get d_fth/dW
        z_dot is a function of x, y, w, AKA df/dw 
        """
        if W.dim() == 0:
            W = W.expand(z.shape[0], 1)
        else:
            W = W.reshape(z.shape[0], 1)
        Wz = torch.cat((W, z), 1)
        z_dot = self.f(Wz)
        return z_dot    

    def getGrads(self, Wz, getJerk = False):
        """
        Wz: N (d+1)
        out: N d
        jacs:
        """
        Wz.requires_grad_(True)
        N = Wz.shape[0]
        dim = Wz.shape[1]-1 # dimension
        batchsize = Wz.shape[0]
        z = Wz[:, 1:]
        W = Wz[:, :1]
        out = self.get_z_dot(W, z) #df/dw

        #jacobians = derivative 
        #in my implementation: gets the derivative of f_th wrt b, and df_dw_db 
        jacobians = torch.zeros(batchsize, dim, dim+1).to(Wz) #f_th/dw
        for i in range(dim):
            jacobians[:, i, :] = torch.autograd.grad(
                out[:, i].sum(), Wz, create_graph=True)[0] #gets df/db

        df_dw_db = torch.zeros(batchsize, dim, dim+1).to(Wz) #f_th/dw
        for i in range(dim):
            df_dw_db[:, i, :] = torch.autograd.grad(out[:, i].sum(), Wz, create_graph=True)[0] #gets df/db
            

        # get Jerk. 3rd time deriv. Promotes constant rotation.
        Jerk = torch.zeros(N,dim)
        if getJerk:
            for i in range(dim):
                JerkTZ = torch.autograd.grad(jacobians[:, i, dim].sum(), Wz, create_graph=True)[0]
                Jerk[:,i] = JerkTZ[:,dim]

        return out, jacobians[:, :, 0:dim], jacobians[:, :, dim:], Jerk

    def forward(self, t, z):
        """
        Calculate the time derivative of z.
        Parameters
        ----------
        t : torch.Tensor
            time
        z : torch.Tensor
            state
        W : torch.Tensor
            weight
        Returns
        -------
        z_dot : torch.Tensor
            Time derivative of z.
            W derivative of z = f(W)
        """
        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
            W.requires_grad_(True)
            z_dot = self.get_z_dot(t, z)
            
        return z_dot

    def forward(self, xi):
        return self.f(xi) # self.f(xi) 

img = np.array()
train_set = np.array()
test_set = np.array()

subnetwork = InputMapping()
W = list(subnetwork.parameters())
f_theta = velocMLP()

pred = f_theta(train_set)
pred = f_th(W*sin(B*x) + w*sin(b*x))

def loss(y, pred): 
    L_result = y-pred
    return L_result 

L = loss(y, pred)

# L = y-f_th(W*sin(B*x) + w*sin(b*x))
# external_grad = torch.tensor([1., 1.])
# L.backward(gradient=external_grad)
# L.backward()
dl_dw = w.grad
dl_dw.backward()
dl_dw_db = dl_dw.grad

learning_rate = 1e-3
epochs = 5
batch_size = 64
loss_fn = (y - f_th(Wsin(Bx) + wsin(bx)))**2

def E(nums): #expected value func
    #torch.nn.functional.binary_cross_entropy(input, target, weight=None, size_average=None, reduce=None, reduction='mean')
    return np.mean(nums)
#first W
W = Variable(torch.randn(), require_grad = True)
b = Variable(torch.rand, requires_grad = True)
optimizer = optim.Adam([W, b])

for sample, target in zip(data, targets): 
    optimizer.zero_grad()
    pred = f_theta(train_set)
    loss = pred - Y
    loss.backward()
    optimizer.step()
#first w: 
w = 0
w = np.zeroes()
w = 1 

def first_b_guess():
L = (E[y- f_th(Wsin(Bx) + wsin(bx))])**2 
L = E(y-f_theta())

f_th = 0
d_fth_dW = 0
d_fth_dW_db = 0 
d_fth_db = 0

dL_dw = -2*(y - E[f_th(Wsin(Bx) + wsin(bx))*f_th'(Wsin(Bx) + wsin(bx))*sin(bx)])
            = -2*(y - E[f_th*f_th'*sin(bx)] #f_th' is wrt w
        = -2*(y - f_th*df_th_dW*sin(bx))

dL_dw_db = 2*E[f_th*f_th'*b*cos(bx) + sin(bx)*(f_th'*f_th'*wbcos(bx) + f_th*f_th''*wbcos(bx))]
                  = 2*(E[f_th*f_th'*b*cos(bx)] + E[sin(bx)*(f_th'*f_th'*wbcos(bx) + f_th*f_th''*wbcos(bx))])
        = 2*f_th*df_th_dw*b*cos(bx) + sin(bx)*(df_th_db*w*b*cos(bx) + f_th*df_dW_db*w*b*cos(bx))

try values of b until dL_dw_db = 0 
or set 0 = dL_dw_db 
or use gradient descent? using function optim_fn(parameters, lr = )
find b, which (hopefully) is the max dL/dw

b_opt = 0
B = B.append(b)

class Losses():
    def __init(self):
        super().__init__()
    def 


#MAKE A GITHUB

#optimization has 3 steps: 
Call optimizer.zero_grad() to reset the gradients of model parameters. Gradients by default add up; to prevent double-counting, we explicitly zero them at each iteration.
Backpropagate the prediction loss with a call to loss.backward(). PyTorch deposits the gradients of the loss w.r.t. each parameter.
Once we have our gradients, we call optimizer.step() to adjust the parameters by the gradients collected in the backward pass.


class MyLoss(Function):
    def forward(ctx, y_pred, y):    
        """In the forward pass we receive a Tensor containing the input and return a
        Tensor containing the output. You can cache arbitrary Tensors for use in the
        backward pass using the save_for_backward method.
        """
        ctx.save_for_backward(y_pred, y)
        return ((y - y_pred)**2).mean()

    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        defines gradient formula
        It will be given as many Tensor arguments as there were outputs, with each of them representing gradient w.r.t. that output. 
        outputs: w, b? or just b?"""

        #y_pred is the same as f_th
        y_pred, y = ctx.saved_tensors
        d_fth_dW = get_z_dot(W, )

        #gradient of loss wrt input 
        dL_dw = -2*(y - y_pred*df_th_dW*sin(bx)) 
        grad_input = 2 * (y_pred - y) / y_pred.shape[0]    

        dL_dw_db = 2*f_th*df_th_dw*b*cos(bx) + sin(bx)*(df_th_db*w*b*cos(bx) + f_th*df_dW_db*w*b*cos(bx))
        
        return grad_input, None

#gradient descent 

W = Variable(torch.randn(4, 3), requires_grad=True)
b = Variable(torch.randn(3), requires_grad=True)

for sample, target in zip(data, targets):
    # clear out the gradients of Variables 
    # (i.e. W, b)
    W.grad.data.zero_()
    b.grad.data.zero_()

    output = linear_model(sample, W, b)
    loss = (output - target) ** 2
    loss.backward() #sum of gradients happens 

    W -= learning_rate * W.grad.data
    b -= learning_rate * b.grad.data

W = Variable(torch.randn(4, 3), requires_grad=True)
b = Variable(torch.randn(3), requires_grad=True)

for sample, target in zip(data, targets):
    # clear out the gradients of all Variables 
    # in this optimizer (i.e. W, b)
    optimizer.zero_grad()
    output = linear_model(sample, W, b)
    loss = (output - target) ** 2
    loss.backward()
    optimizer.step()