from tkinter import Y
import torch
from torch import nn 
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm as tqdm
import os, imageio


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class InputMapping(nn.Module):
    """Fourier features mapping
    contains W layer 
    implements wsinbx"""

    #should B be an input parameter in InputMapping?
    def __init__(self, d_in, n_freq, hidden_features, sigma=2, tdiv=2,  Tperiod=None): 
        # removed parameter: incrementalMask=True, added hiddenfeatures
        super(InputMapping, self).__init__()
        Bmat = torch.randn(n_freq, d_in) * np.pi* sigma/np.sqrt(d_in)  # gaussian
        # B_first_guess = 

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
        print(n_freq)
        self.n_freq = n_freq
        self.d_out = n_freq * 2 + d_in
        print('in B')
        print(d_in, self.d_out)
        self.B = nn.Linear(2, 2, bias=False)
        
        self.W = nn.Linear(2, hidden_features)
        # with torch.no_grad():
        #     self.B.weight = nn.Parameter(Bmat.to(device), requires_grad=False)
        #     self.mask = nn.Parameter(torch.zeros(
        #         1, n_freq), requires_grad=False)

        # self.incrementalMask = incrementalMask
        # if not incrementalMask:

        # self.mask = nn.Parameter(torch.ones(
        #         1, n_freq), requires_grad=False)
        self.mask = nn.Parameter(torch.ones(
                1, n_freq), requires_grad=False)

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

    def forward(self, xi, hidden_features=512):
        # pdb.set_trace()
        # dim = xi.shape[1]-1
        dim = 2
        print('shape, ',xi.shape)
        print(xi)
        xi = torch.from_numpy(xi).float()
        y = self.B(xi)
        print('won')
        # if self.Tperiod is None:
        #     net = torch.cat([torch.sin(y)*self.mask, torch.cos(y)*self.mask, xi], dim=-1)
        # else:
        #     net = torch.cat([torch.sin(y)*self.mask, torch.cos(y)*self.mask, xi[:,1:dim+1]], dim=-1)
        
        net = torch.cat([torch.sin(y)*self.mask, xi], dim=-1)
        # net = torch.sin(y)*self.mask
        net.append(nn.Linear(self.d_out), hidden_features) #W layer
        # net.append(torch.ones(256,1))
        # X = np.matmul(net, self.W)
        return net

# class velocMLP(nn.Module): 
class model(nn.Module): 
    """
    implements f_th()
    """
    def __init__(self, in_features=3, hidden_features=512, hidden_layers=2,
                 out_features=2, sigmac=3, n_freq=2, tdiv=1,
                  Tperiod=None): #removed param incrementalMask=True,
        super(model, self).__init__()
        print('in model')
        net = []
        imap = InputMapping(in_features, n_freq, hidden_features, sigma=sigmac,
                            tdiv=tdiv, Tperiod=Tperiod)
        print('in imap')
        self.imap = imap
        net.append(imap)

        # deleted linear layer W, put in InputMapping
        # net.append(nn.Linear(imap.d_out, hidden_features)) 
        for i in range(hidden_layers):
            print('in hidden')
            net.append(nn.Tanh())
            net.append(nn.Linear(hidden_features, hidden_features))
        net.append(nn.Softplus())
        net.append(nn.Linear(hidden_features, out_features))
        net = nn.Sequential(*net)
        net = net.float()
        self.f = net
        print('done with net')

    def df_dw_func(self, W, x):
        """
        gets df/dw
        z_dot is parameterized by a NN: z_dot = NN(t, z(t))
        z_dot is the velocity, and is a function of x, y, t (not a constant value)
        df/dw is a function of (x, y) coordinate, W
        replace t with W to get d_fth/dW
        z_dot is a function of x, y, w, AKA df/dw 
        z is (x, y) aka x aka coord
        """
        print('df_dw func')
        if W.dim() == 0:
            W = W.expand(x.shape[0], 1)
        else:
            W = W.reshape(x.shape[0], 1)
        Wx = torch.cat((W, x), 1)
        df_dW = self.f(Wx)
        return df_dW

    def get_z_dot(self, W, z):
        """
        gets df/dw
        z_dot is parameterized by a NN: z_dot = NN(t, z(t))
        z_dot is the velocity, and is a function of x, y, t (not a constant value)
        df/dw is a function of (x, y) coordinate, W
        replace t with W to get d_fth/dW
        z_dot is a function of x, y, w, AKA df/dw 
        z is (x, y) aka x aka coord
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

    # def forward(self, t, z):
    #     """
    #     Calculate the time derivative of z.
    #     Parameters
    #     ----------
    #     t : torch.Tensor
    #         time
    #     z : torch.Tensor
    #         state
    #     W : torch.Tensor
    #         weight
    #     Returns
    #     -------
    #     z_dot : torch.Tensor
    #         Time derivative of z.
    #         W derivative of z = f(W)
    #     """
    #     with torch.set_grad_enabled(True):
    #         z.requires_grad_(True)
    #         W.requires_grad_(True)
    #         z_dot = self.get_z_dot(t, z)
            
    #     return z_dot
    def get_parameter(self, target: str) -> "Parameter":
        return super().get_parameter(target)

    def forward(self, xi):
        print('in forward')
        return self.f(xi) 

#image data
# Download image, take a square crop from the center
image_url = 'https://live.staticflickr.com/7492/15677707699_d9d67acf9d_b.jpg'
img = imageio.imread(image_url)[..., :3] / 255.
c = [img.shape[0]//2, img.shape[1]//2]
r = 256
img = img[c[0]-r:c[0]+r, c[1]-r:c[1]+r]
# plt.imshow(img)
# plt.show()

# Create input pixel coordinates in the unit square
coords = np.linspace(0, 1, img.shape[0], endpoint=False)
x_test = np.stack(np.meshgrid(coords, coords), -1)
test_data = [x_test, img]
# train_data = [x_test[::2,::2], img[::2,::2]]

train_data = np.zeros((256,2))
# for i in range(256):
#     train_data[i] = np.array([x_test[::2,::2][i], img[::2,::2][i]])
# # input_model = InputMapping(51)
my_model = model()
my_model = my_model.float()
train_data = np.concatenate((x_test[::2,::2], img[::2,::2]), axis=2)
# print(train_data.shape)
train_data = np.reshape(train_data, (256*256, 5))
# optimizer = torch.optim.Adam(W, lr=0.0001)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# def get_parameters(model, bias=False):
#     for k, m in model._modules.items():
#         print("get_parameters", k, type(m), type(m).__name__, bias)
#         if bias:
#             if isinstance(m, nn.Conv2d):
#                 yield m.bias
#         else:
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
#                 yield m.weight

# def get_parameters(model):
#     for w in model

# optimizer = torch.optim.Adam(
#             [
#                 {'params': get_parameters(model, bias=False)},
#                 {'params': get_parameters(model, bias=True)},
#             ],
#             lr=0.0001)
optimizer = torch.optim.Adam(
            [param for param in my_model.parameters()],
            lr=0.0001)
            
# optimizer = torch.optim.Adam(lr = 0.00001)
#train model
train_loss_array = []
# for sample, target in zip(data, targets): 
counter = 0
# sample = train_data[:2]
# target = train_data[2:]
for i in train_data: 
    sample = np.array(i[:2])
    target = np.array(i[2:])
    sample = np.array([sample[0].astype(float), sample[1].astype(float)])
    target = target.astype(float)
    # sample, target = i[:2], i[2:]
    print(sample, target)
    print(sample.shape, target.shape)
    #dL_dw_func = lambda y, pred, W, x, b: -2*(y - pred*df_dW(W, x)*sin(bx))
    x = sample
    y = target
    pred = my_model(sample)
    W = torch.tensor.zeros()
    dL_dw_func = lambda b: -2*(y - pred*my_model.df_dw_func(W, x)*np.sin(b*x))
    # def dL_dw(b_guess): 
    #     return -2*(y - pred*model.df_dW(W, x)*sin(bx))
    list_b = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    dl_dws = [dL_dw_func(b_guess) for b_guess in list_b]
    best_guess_slope = min(dl_dws)
    best_b_guess = np.argmin(dl_dws)
    
    # clear out the gradients of all Variables 
    # in this optimizer (i.e. W, b)
    optimizer.zero_grad()

    print('counter:', counter)
    # feed our input through the network
    # subnetwork = InputMapping(sample)
    # output = linear_model(sample, W, b)
    # pred = f_theta(train_set)
    pred = my_model.forward(sample)
    print('done with model')
    #calculate loss
    y = target
    x = sample
    loss = pred - y
    train_loss_array.append(loss)

    #backpropogation 
    #updating weights by taking derivative of the loss function 
    loss.backward()

    
    # getGrads(W)
    #updtes the value of b using the gradient b.grad?
    #ex: SGD does x += -lr * x.grad

    #change w's 
    optimizer.step()
    counter += 1
print(train_loss_array)

#test model 
test_loss_array = []
for sample, target in test_data: 
    pred = model(sample)
    loss = target - pred 
    test_loss_array.append(loss)

print(test_loss_array)

# for epoch in range(1, epochs+1):
#     train_loss = train(deep_autoencoder, device, train_loader, optimizer)
#     val_loss = test(deep_autoencoder, device, val_loader)

# subnetwork = InputMapping(sample)
# W = list(subnetwork.parameters())
# f_theta = velocMLP(W)
# pred = f_theta(train_set)
# # pred = f_th(W*sin(B*x) + w*sin(b*x))
# L = loss(target, pred)

learning_rate = 1e-3
epochs = 5
batch_size = 64
# loss_fn = (y - f_th(Wsin(Bx) + wsin(bx)))**2

# #first W
# W = Variable(torch.randn(), require_grad = True)
# b = Variable(torch.rand, requires_grad = True)
# optimizer = optim.Adam([W, b])

# for sample, target in zip(data, targets): 
#     optimizer.zero_grad()
#     pred = f_theta(train_set)
#     loss = pred - Y
#     loss.backward()
#     optimizer.step()
# #first w: 
# w = 0
# w = np.zeroes()
# w = 1 

# f_th = 0
# d_fth_dW = 0
# d_fth_dW_db = 0 
# d_fth_db = 0

# dL_dw = -2*(y - E[f_th(Wsin(Bx) + wsin(bx))*f_th'(Wsin(Bx) + wsin(bx))*sin(bx)])
#             = -2*(y - E[f_th*f_th'*sin(bx)] #f_th' is wrt w
#         = -2*(y - f_th*df_th_dW*sin(bx))
        

# dL_dw_db = 2*E[f_th*f_th'*b*cos(bx) + sin(bx)*(f_th'*f_th'*wbcos(bx) + f_th*f_th''*wbcos(bx))]
#                   = 2*(E[f_th*f_th'*b*cos(bx)] + E[sin(bx)*(f_th'*f_th'*wbcos(bx) + f_th*f_th''*wbcos(bx))])
#         = 2*f_th*df_th_dw*b*cos(bx) + sin(bx)*(df_th_db*w*b*cos(bx) + f_th*df_dW_db*w*b*cos(bx))

# try values of b until dL_dw_db = 0 
# or set 0 = dL_dw_db 
# or use gradient descent? using function optim_fn(parameters, lr = )
# find b, which (hopefully) is the max dL/dw

# b_opt = 0
# B = B.append(b)

# class Losses():
#     def __init(self):
#         super().__init__()
#     def 


# #optimization has 3 steps: 
# Call optimizer.zero_grad() to reset the gradients of model parameters. Gradients by default add up; to prevent double-counting, we explicitly zero them at each iteration.
# Backpropagate the prediction loss with a call to loss.backward(). PyTorch deposits the gradients of the loss w.r.t. each parameter.
# Once we have our gradients, we call optimizer.step() to adjust the parameters by the gradients collected in the backward pass.

# class MyLoss(Function):
#     def forward(ctx, y_pred, y):    
#         """In the forward pass we receive a Tensor containing the input and return a
#         Tensor containing the output. You can cache arbitrary Tensors for use in the
#         backward pass using the save_for_backward method.
#         """
#         ctx.save_for_backward(y_pred, y)
#         return ((y - y_pred)**2).mean()

#     # bias is an optional argument
#     def forward(ctx, input, weight, bias=None):
#         ctx.save_for_backward(input, weight, bias)
#         output = input.mm(weight.t())
#         if bias is not None:
#             output += bias.unsqueeze(0).expand_as(output)
#         return output

#     def backward(ctx, grad_output):
#         """
#         In the backward pass we receive a Tensor containing the gradient of the loss
#         with respect to the output, and we need to compute the gradient of the loss
#         with respect to the input.

#         defines gradient formula
#         It will be given as many Tensor arguments as there were outputs, with each of them representing gradient w.r.t. that output. 
#         outputs: w, b? or just b?"""

#         #y_pred is the same as f_th
#         y_pred, y = ctx.saved_tensors
#         d_fth_dW = get_z_dot(W, )
#         df_dw_func = model.df_dw_func

#         #gradient of loss wrt input 
#         dL_dw = -2*(y - y_pred*df_th_dW*sin(bx)) 
#         grad_input = 2 * (y_pred - y) / y_pred.shape[0]    

#         dL_dw_db = 2*f_th*df_th_dw*b*cos(bx) + sin(bx)*(df_th_db*w*b*cos(bx) + f_th*df_dW_db*w*b*cos(bx))
        
#         return grad_input, None

# #gradient descent 
# W = Variable(torch.randn(4, 3), requires_grad=True)
# b = Variable(torch.randn(3), requires_grad=True)

# for sample, target in zip(data, targets):
#     # clear out the gradients of Variables 
#     # (i.e. W, b)
#     W.grad.data.zero_()
#     b.grad.data.zero_()

#     output = linear_model(sample, W, b)
#     loss = (output - target) ** 2
#     loss.backward() #sum of gradients happens 

#     W -= learning_rate * W.grad.data
#     b -= learning_rate * b.grad.data

# W = Variable(torch.randn(4, 3), requires_grad=True)
# b = Variable(torch.randn(3), requires_grad=True)


# for e in range(100):
#     loss = abs(get_z_dot(W, b) - 0) #get z dot is dL_dw(b)
#     optimizer.zero_grad()
#     answer = get_z_dot(W, b

# list_b = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# calculate dL_dW(b): find the max or min of this list? 
# then make small increments around the b value 
# dl_dW_list = [dL_dW(b_guess) for b_guess in list_b]
# max_index = maxindex(dl_dW_list)

# dl_dws = [dL_dW(b_guess) for b_guess in list_b]
# best_guess_slope = min(dl_dws)
# best_b_guess = np.argmin(values)

# train_loss_array = []

# dL_dw = -2*(y - f_th*df_th_dW*sin(bx))
# dL_dw = -2*(y - output*df_dW(W, z)*sin(bx))
# def dL_dw(y, output, b_guess, x):
#     return -2*(y - output*df_dW(W, x)*sin(bx))

# list_b = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# # calculate dL_dW(b): find the max or min of this list? 
# # then make small increments around the b value 
# dl_dW_list = [dL_dW(b_guess) for b_guess in list_b]
# max_index = maxindex(dl_dW_list)

# dl_dws = [dL_dW(b_guess, x) for b_guess in list_b]
# best_guess_slope = min(dl_dws)
# best_b_guess = np.argmin(values)
# # def find_best_b_guess():

