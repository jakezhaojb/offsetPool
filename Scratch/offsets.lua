require 'cunn' 
require 'jzt'
require 'xlua' 

cutorch.setDevice(2); 

bsz = 2
inplane = 2 

kW = 3
kH = 3 
W = 8
H = 8

m = jzt.SSMPoolingOffsets(kH,kW):cuda() 
--m = nn.SpatialConvolution(inplane,inplane,kH,kW):cuda() 
--m = nn.SpatialLPPooling(inplane,1,kW,kH,dW,dH):cuda() 

input = torch.rand(bsz,inplane,H,W):cuda()

print(testJacobian(m,input))

--err = nn.Jacobian.testJacobian(m,input);

--ox = 2; 
--oy = 2; 
--
--for ii = 1,bsz do
-- for jj = 1,inplane do
--  for yi = 1,H,kH do 
--   for xi = 1,W,kW do 
--    if yi+oy<=H and xi+ox<=W then 
--        input[ii][jj][yi+oy][xi+ox] = 10;
--    end
--   end
--  end
-- end
--end

out = m:forward(input)
gradOut = torch.ones(out:size()):mul(-1):cuda()
grad = m:backward(input,gradOut)


--
--out = m:forward(x)
--
--print(out) 

