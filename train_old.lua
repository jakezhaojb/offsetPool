dofile('init.lua')
math.randomseed(1)
torch.manualSeed(1) 
cutorch.manualSeed(1) 
torch.setdefaulttensortype('torch.FloatTensor')
if math.mod == nil then
  math.mod = math.fmod
end

cutorch.setDevice(1) 

im = image.lena()
im = image.scale(im,198,198)
im = padImage(im, {1,1,1,1}, 0)

im = image.rgb2y(im:narrow(1,1,3)):clone() 
im = padImage(im, {1, 1, 1, 1}, 0)  -- image, pad_lrtb, bordervalue
w = im:size(3)
h = im:size(2)

plot_sample = function(sample, net, w, h, win) 
   
   --plot a sample 
    local im = sample.images[{{1,16},{},{},{}}] 
    local loc_out = net:forward(sample.images) 
    loc_out:select(2,1):mul(w) 
    loc_out:select(2,2):mul(h) 
    local loc_true = sample.targets:clone() 
    loc_true:select(2,1):mul(w) 
    loc_true:select(2,2):mul(h) 
    im = im:float():repeatTensor(1,3,1,1) 
    
    for i = 1,im:size(1) do 
      drawLabels(im[i], loc_true[i], w, h, 1)
      drawLabels(im[i], loc_out[i] , w, h, 0, 0.02)
    end
   
    --if win then 
    --    win = image.display{image=im, padding=1, nrow=8, zoom=1, win=win}
    --else 
    --    win = image.display{image=im, padding=1, nrow=8, zoom=1}
    --end 
    im = image.toDisplayTensor({input=im,nrow=8,padding=1}) 

    return im 
end

get_batch = function(batch, bsz, im) 

    local h = im:size(2) 
    local w = im:size(3) 
    local scale = 0.5 
    local uv = torch.FloatTensor(2, bsz)
    uv[{1,{}}]:fill(w/2)
    uv[{2,{}}]:fill(h/2)

    for i = 1,bsz do
      
      local deg_rot = math.random() * 360 
      local trans_u = (math.random() - 0.5) * w
      local trans_v = (math.random() - 0.5) * h
    
      --local 
      rand_im = distortImage{im=im, deg_rot=deg_rot, scale=scale, trans_u_pix=trans_u, trans_v_pix = trans_v}

      batch.images[i]:copy(rand_im)
      distortPos2D(uv[{{},{i}}], deg_rot, scale, trans_v, trans_u, w, h)
    
    end
   
    --subtract the mean 
    batch.images:add(-batch.images:mean())
    --set targets from 0->1 
    uv[1]:div(w)
    uv[2]:div(h) 
    batch.targets:copy(uv:t())  
    return batch 

end
    
train_sample = function(net, sample, learn_rate, criterion)    
   
    local output = net:forward(sample.images) 
    
    -- Calculate the pixel error
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

test_error = function(dataset,net,criterion) 

    local out = transform_data(dataset.images, net) 
    return criterion:forward(dataset.targets,out) 

end 

--==Hyper Parameters==
bsz = 16 
learn_rate = 0.0001
nsamples = 100  -- number of batches in an epoch
epochs = 100 
pool1 = 40
pool2 = 1
phase_const = 0
mag_const = 1
net = phase_net1(im:size(),pool1,phase_const,mag_const):cuda() 
--====================
sample = {}
sample.images = torch.CudaTensor(bsz,1,h,w)  
sample.targets = torch.CudaTensor(bsz,2) 
test_sample = {}
test_sample.images = torch.CudaTensor(bsz,1,h,w)  
test_sample.targets = torch.CudaTensor(bsz,2) 
get_batch(test_sample,bsz,im) 
criterion = nn.MSECriterion():cuda() 
criterion.sizeAverage = true

for iter = 1,epochs do 

    epoch_train_error = 0
    sys.tic() 
    local uv_errors = torch.FloatTensor(nsamples, bsz)
    for i = 1, nsamples do
        
        if sample_train_error ~= nil then
           progress(i,nsamples,string.format('err=%.4e', sample_train_error))
        else
           progress(i,nsamples)
        end
        get_batch(sample,bsz,im) 
        sample_train_error, sample_train_error_pix = 
           train_sample(net, sample, learn_rate, criterion)

        uv_errors[{i,{}}]:copy(sample_train_error_pix)

        epoch_train_error = epoch_train_error + sample_train_error 
    end
    time = sys.toc() 

    av_epoch_train_error = epoch_train_error / nsamples
    local output = iter..'-Time: '..time..' Train Error: '..
    av_epoch_train_error..', '..uv_errors:mean()..'(pix)'
    
    uv_errors = uv_errors:reshape(nsamples * bsz)
    
    gnuplot.hist(uv_errors, 50, 0, 100)
    gnuplot.figprint('./Results/err_hist_iter_pool'..(pool1*pool2)..'.eps')
    --gnuplot.closeall()

    print(output) 
    sample_im = plot_sample(test_sample,net,w,h) 

    image.save('./Results/sample.png',sample_im) 

end  
