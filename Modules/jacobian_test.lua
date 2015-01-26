torch.setdefaulttensortype('torch.DoubleTensor')

dofile('init.lua')
cutorch.setDevice(2) 

bsz = 3 
inplane = 2
W = 8
H = 9
kW = 2
kH = 2 

if jzt == nil then
  jzt = nn
end

--cpu version 
module_cpu = jzt.SSMPoolingOffsets(kW,kH)
input_cpu = torch.rand(bsz,inplane,H,W) 
output_cpu = module_cpu:forward(input_cpu)
gradOutput_cpu = torch.ones(output_cpu:size())
gradInput_cpu = module_cpu:backward(input_cpu, gradOutput_cpu) 

--gpu version 
module_gpu = module_cpu:clone():cuda() 
input_gpu = input_cpu:cuda()  
output_gpu = module_gpu:forward(input_gpu)
gradOutput_gpu = gradOutput_cpu:cuda() 
gradInput_gpu = module_gpu:backward(input_gpu, gradOutput_gpu) 

--check 
err_fprop = output_cpu:clone():add(-output_gpu:double()):max() 
err_bprop = gradInput_cpu:clone():add(-gradInput_gpu:double()):max()

print('fprop error: '..err_fprop) 
print('bprop error: '..err_bprop)

--Jacobian test 
torch.setnumthreads(8)

-- Create an instance of the test framework
local precision = 1e-5
local mytester = torch.Tester()
local jac = nn.Jacobian
local test = {}

function test.OffsetPooling()
  local input = input_cpu:clone() 
  local module = module_cpu:clone() 

  local err = jac.testJacobian(module,input)
  print(err)
  mytester:assertlt(err, precision, 'error on state ')

  local ferr,berr = jac.testIO(module,input)
  mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
  mytester:asserteq(berr, 0, torch.typename(module) .. ' - i/o backward err ') 
end

function test.OffsetPoolingGPU()
  local input = input_gpu:clone()
  local mod = module_gpu:clone()
  local mod2 = module_cpu:clone()

  -- Test FPROP
  local output = mod:forward(input:cuda()):double()
  local output2 = mod2:forward(input:double())
  mytester:assertlt((output - output2):abs():max(), precision, 'fprop error ')

  -- Test BPROP
  local gradOutput = gradOutput_gpu
  local gInput = mod:backward(input:cuda(), gradOutput:cuda()):double()
  local gInput2 = mod2:backward(input:double(), gradOutput:double())
  mytester:assertlt((gInput - gInput2):abs():max(), precision, 'bprop error ')
end

-- Now run the test above
mytester:add(test)
mytester:run()



