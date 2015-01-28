dofile("init.lua")
cutorch.setDevice(2)

train_sample = function(sample, net, criterion, w, h)
   local output = net:forward(sample.images)
   -- pixel error
   local out_pix = output:float()
   out_pix[{{},1}]:mul(w)
   out_pix[{{},2}]:mul(h)
   if out_pix:size(2) == 4 then
      out_pix[{{},3}]:mul(w)
      out_pix[{{},4}]:mul(h)
   end
   local out_target = sample.targets:float()
   out_target[{{},1}]:mul(w)
   out_target[{{},2}]:mul(h)
   if out_target:size(2) == 4 then
      out_target[{{},3}]:mul(w)
      out_target[{{},4}]:mul(h)
   end
   local pix_err = out_pix - out_target
   pix_err = pix_err:pow(2):sum(2):sqrt()  -- mean of L2 on dim 1

   local sample_error = criterion:forward(output,sample.targets) 
   local gradOutput = criterion:backward(output,sample.targets) 
   
   --learning 
   net:zeroGradParameters() 
   net:backward(sample.images,gradOutput) 
   net:updateParameters(learn_rate) 

   return sample_error, pix_err
end

dofile("init.lua")

local train_data_path = "./toy_dataset_child/toy.t7"
if not paths.filep(train_data_path) then
   local dir_path = paths.dirname(train_data_path)
   os.execute("cd " .. dir_path .. "&& ~/install/new_torch7_2014_10_24/bin/th generate_toy_dataset.lua")
end

dataset = torch.load( "./toy_dataset_child/toy.t7")

bsz = 16
learn_rate = 0.0001
nbatches = 100  -- number of batches in an epoch
epochs = 200 
--pool1 = 40 -- TODO
pool2 = 2
phase_const = 0
mag_const = 1
imsz = dataset.images[1]:size() 
width = imsz[2]
height = imsz[3]
pool1 = math.floor(width / 5)
if dataset.targets:float():size(2) == 4 then
   net = phase_net2(imsz,pool1,pool2,phase_const,mag_const,true):cuda()  --TODO
else
   net = phase_net2(imsz,pool1,pool2,phase_const,mag_const):cuda()  --TODO
end
--====================
--[[
sample = {}
sample.images = torch.CudaTensor(bsz,1,h,w)  
sample.targets = torch.CudaTensor(bsz,2) 
test_sample = {}
test_sample.images = torch.CudaTensor(bsz,1,h,w)  
test_sample.targets = torch.CudaTensor(bsz,2) 
get_batch(test_sample,bsz,im) 
--]]
criterion = nn.MSECriterion():cuda() 
criterion.sizeAverage = true

local numTrainDataset = dataset.targets:size(1)
nbatches = math.floor(numTrainDataset / bsz)

--dataset.images = dataset.images:cuda()
--dataset.targets = dataset.targets:cuda()

for iter = 1,epochs do 

    idx = torch.randperm(numTrainDataset)
    epoch_train_error = 0
    sys.tic() 
    local uv_errors = torch.FloatTensor(nbatches, bsz)
    uv_errors:zero()

    for i = 1, nbatches do
        idx_batch = (i-1) * bsz + 1
        samples = {}
        samples.targets = dataset.targets[{ {idx_batch, idx_batch+bsz-1}, {} }]
        samples.images = dataset.images[{ {idx_batch, idx_batch+bsz-1},{},{},{} }]
        samples.targets = samples.targets:cuda()
        samples.images = samples.images:cuda()
        
        if sample_train_error ~= nil then
           progress(i,nbatches,string.format('err=%.4e', sample_train_error))
        else
           progress(i,nbatches)
        end

        sample_train_error, sample_train_error_pix = 
           train_sample(samples, net, criterion, width, height)

        uv_errors[{i,{}}]:copy(sample_train_error_pix)

        epoch_train_error = epoch_train_error + sample_train_error 

        samples.targets = nil
        samples.images = nil

        collectgarbage()
    end
    time = sys.toc() 

    av_epoch_train_error = epoch_train_error / nbatches
    local output = iter..'-Time: '..time..' Train Error: '..
    av_epoch_train_error..', '..uv_errors:mean()..'(pix)'
    
    uv_errors = uv_errors:reshape(nbatches * bsz)
    
    gnuplot.hist(uv_errors, 50, 0, 100)
    gnuplot.figprint('./Results/phase2_child_err_hist_iter_pool'..(pool1*pool2)..'.eps')
    --gnuplot.closeall()

    print(output) 
end  

torch.save("./Results/phase2_child.net", net)
