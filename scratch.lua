dofile('init.lua')

cutorch.setDevice(2)

bsz = 1
inplane = 1
W = 48
H = 48
kW = 4
kH = 4 
gpu = true 

if gpu == true then 
    print('gpu') 
    module = jzt.SSMPoolingOffsets(kW,kH):cuda()
    input = torch.rand(1,W*H):reshape(1,1,H,W):repeatTensor(bsz,inplane,1,1):cuda() 
    output = module:forward(input)
    gradOutput = torch.zeros(output:size()):cuda() 
    gradInput = module:backward(input, gradOutput) 
    print(gradInput:max())
    collectgarbage()
else 
    print('cpu') 
    module = jzt.SSMPoolingOffsets(kW,kH)
    input = torch.rand(1,W*H):reshape(1,1,H,W):repeatTensor(bsz,inplane,1,1)
    output = module:forward(input)
    gradOutput = torch.zeros(output:size())
    gradInput = module:backward(input, gradOutput) 
    print(gradInput:max())
    collectgarbage()
end


