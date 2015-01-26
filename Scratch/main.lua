require 'cunn' 
require 'xlua' 
--dofile('SpatialSoftMax.lua')

beta = 0.5
bsz = 16
inplane = 3
inx = 8 
iny = 4 
px = 2 
py = 2
outx= math.floor(inx/px) 
outy= math.floor(iny/py) 
x = torch.range(1,inx*iny):resize(1,1,iny,inx);
x = x:repeatTensor(bsz,inplane,1,1):cuda()  


input = x:clone() 
input = input:resize(bsz*inplane,iny,inx) 
--self = {} 
--self.kH = py 
--self.kW = px 
--self.beta = beta 
--self.output = torch.Tensor() 
--self.norm = torch.Tensor() 
--
--    local kW = self.kW 
--    local kH = self.kH
--    local bsz = input:size(1) 
--    local inplane = input:size(2) 
--    local inW = input:size(3) 
--    local inH = input:size(4)
--    local outW = math.floor(inW/kW) 
--    local outH = math.floor(inH/kH) 
--
--    self.output = self.output:typeAs(input):resize(bsz*inplane,inH,inW) 
--    self.norm = self.norm:typeAs(input):resize(bsz*inplane,inH,inW) 
--    self.output:copy(input) 
--    self.output:mul(self.beta):exp() 
--    
--    self.norm:copy(self.output:resize(bsz*inplane,inH,inW)) 
--    self.norm = self.norm:unfold(2,kH,kH):unfold(3,kW,kW):contiguous():resize(bsz*inplane,outy,outx,py*px)
--    self.norm = self.norm:sum(4):resize(bsz*inplane,outH,outW,1,1) 
--    self.norm = self.norm:expand(bsz*inplane,outH,outW,kH,kW):transpose(3,4):contiguous():resize(bsz*inplane,inH,inW)
--    self.output:cdiv(self.norm)
--    self.output:resizeAs(input)
--    self.norm:resizeAs(input) 


out = input:clone():mul(beta):exp()
n=10000

sys.tic() 
for i = 1,n do 
norm = out:clone()
norm = norm:unfold(2,py,py):unfold(3,px,px):contiguous():resize(bsz*inplane,outy,outx,py*px)
norm = norm:sum(4):resize(bsz*inplane,outy,outx,1,1)
norm = norm:expand(bsz*inplane,outy,outx,py,px):transpose(3,4):contiguous():resize(bsz*inplane,iny,inx)  
norm1 = norm:clone()
out1 = out:clone()
out1 = out1:cdiv(norm)
out1 = out1:resize(bsz,inplane,iny,inx) 
cutorch.synchronize() 
end
print(sys.toc())

sys.tic() 
for i = 1,n do 
norm = out:clone()
norm = norm:resize(bsz*inplane, outy, py, outx, px):transpose(3,4):contiguous()
norm = norm:resize(bsz*inplane*outy*outx, py*px):sum(2):resize(bsz*inplane, outy, 1, outx, 1)
norm = norm:expand(bsz*inplane, outy, py, outx, px):contiguous():resize(bsz*inplane, outy*py, outx*px)
norm2 = norm:clone()
out2 = out:clone()
out2 = out2:cdiv(norm)
out2 = out2:resize(bsz,inplane,iny,inx) 
cutorch.synchronize() 
end
print(sys.toc())
