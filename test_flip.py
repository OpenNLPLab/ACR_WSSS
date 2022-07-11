import torch

# q = torch.Tensor([[[1,2,3,4,5,6,7,8,9]]])
# k = torch.Tensor([[[2,3,4,5,6,7,8,9,10]]])
# a1 = (q.transpose(1,2) @ k)
# print(a1)


# q = torch.Tensor([[[3,2,1,6,5,4,9,8,7]]])
# k = torch.Tensor([[[4,3,2,7,6,5,10,9,8]]])
# a = (q.transpose(1,2) @ k)
# print(a)
# print(a.shape)

# w=h=3

# for i in range(w):
#     a[:,i*w:i*w+w,:] = a[:,i*w:i*w+w,:].flip(1)
# print(a)
    
# for i in range(w):
#     a[:,:,i*w:i*w+w] = a[:,:,i*w:i*w+w].flip(2)
# print(a)

# print(a1==a)



# print(a.shape)
# a = a.view(1,2,-1).flip(2)
# print(a)
# print(a.shape)

# a = a.view(1,4,-1)
# print(a)
# print(a.shape)

# a = a.view(1,-1,2)
# print(a)
# print(a.shape)

a = torch.rand(2,4,4)
b = torch.rand(2,4,4)

a = (a - torch.amin(a, (1, 2), keepdims=True)) / \
                ( torch.amax(a, (1, 2), keepdims=True) - torch.amin(a, (1, 2), keepdims=True))
            
b = (b - torch.amin(b, (1, 2), keepdims=True)) / \
                ( torch.amax(b, (1, 2), keepdims=True) - torch.amin(b, (1, 2), keepdims=True))

# print(a)
# print(b)
# print(a+b)