dofile('init.lua') 

--cutorch.setDevice(2)

bsz = 1
inplane = 2
W = 8
H = 8

input = torch.rand(bsz,inplane,W,H):mul(1):cuda()

phase = jzt.SSMPoolingOffsets(2,2):cuda()
modulus = nn.SpatialMaxPooling(2,2,2,2):cuda()


offset_pooling = nn.Sequential() 
m = nn.ConcatTable() 
m:add(modulus) 
m:add(phase) 
offset_pooling:add(m) 
offset_pooling:add(nn.JoinTable(2))

out = offset_pooling:forward(input) 


--m2 = nn.SpatialConvolutionFFT(inplane,inplane,2,2,1,1):cuda()
--
--
--out = m1:forward(input)
--grad = m1:backward(input,input:clone())
--
--
--err1 = testJacobian(m1,input);
--
----input = torch.range(1,W*H):reshape(1,1,H,W):repeatTensor(bsz,inplane,1,1)
----input = input:div(input:max()):cuda() 
--
--err2 = testJacobian(m2,input);

--t1 = 0 
--t2 = 0
--
--for ii = 1,1000 do 
--
--    sys.tic() 
--    out = m1:forward(input); 
--    g = m1:backward(input,out:clone())
--    cutorch.synchronize() 
--    t1 = t1 + sys.toc()
--    
--    sys.tic() 
--    out = m2:forward(input); 
--    g = m2:backward(input,out:clone())
--    cutorch.synchronize() 
--    t2 = t2 + sys.toc()
--    
--end
--
--collectgarbage() 
--print(t1)
--print(t2)









