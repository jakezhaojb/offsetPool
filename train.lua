dofile("init.lua")

train_sample = function(sample, net, criterion)
   local output = net:forward(sample.images)
   -- pixel error
   local out_pix = output:float()
   out_pix[{{},1}]:mul(w)
   out_pix[{{},2}]:mul(h)
   local out_target = sample.targets:float()
   out_target[{{},1}]:mul(w)
   out_target[{{},2}]:mul(h)
   local pix_err = out_pix - out_target
   pix_err = pix_err:pow(2):sum(2):sqrt()  -- mean of L2 on dim 2

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
if paths.filep(train_data_path) then
   local dir_path = paths.dirname(train_data_path)
   dofile(dir_path .. "generate_toy_dataset.lua")
end

bsz = 16
learn_rate = 0.0001
nbatches = 100  -- number of batches in an epoch
epochs = 100 
pool1 = 40
pool2 = 1
phase_const = 0
mag_const = 1
net = phase_net1(im:size(),pool1,phase_const,mag_const):cuda()  --TODO
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

for iter = 1,epochs do 

    idx = torch.randperm(numTrainDataset)
    epoch_train_error = 0
    sys.tic() 
    local uv_errors = torch.FloatTensor(nbatches, bsz)

    for i = 1, nbatches do
        idx_batch = (i-1) * bsz + 1
        samples = {}
        samples.targets = dataset.targets[{ {idx_batch, idx_batch+bsz}, {} }]
        samples.images = dataset.images[{ {idx_batch, idx_batch+bsz},{},{},{} }]
        
        if sample_train_error ~= nil then
           progress(i,nbatches,string.format('err=%.4e', sample_train_error))
        else
           progress(i,nbatches)
        end

        sample_train_error, sample_train_error_pix = 
           train_sample(samples, net, criterion)

        uv_errors[{i,{}}]:copy(sample_train_error_pix)

        epoch_train_error = epoch_train_error + sample_train_error 
    end
    time = sys.toc() 

    av_epoch_train_error = epoch_train_error / nbatches
    local output = iter..'-Time: '..time..' Train Error: '..
    av_epoch_train_error..', '..uv_errors:mean()..'(pix)'
    
    uv_errors = uv_errors:reshape(nbatches * bsz)
    
    gnuplot.hist(uv_errors, 50, 0, 100)
    gnuplot.figprint('./Results/err_hist_iter_pool'..(pool1*pool2)..'.eps')
    --gnuplot.closeall()

    print(output) 
end  

torch.save("phase1_child.net", net)
