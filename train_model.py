
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
