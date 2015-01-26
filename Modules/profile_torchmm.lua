require 'xlua'

a = torch.rand(100,100):float()
b = torch.rand(100,100):float()

sys.tic()
for i = 1, 1000 do
  torch.mm(a,b)
end
print(sys.toc())



