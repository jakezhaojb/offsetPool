eval_on_CIFAR = function(trained_layers, learn_rate, epochs, save_path) 

    print('evaluating '..#trained_layers..' trained layers on CIFAR...') 
    ---------------------
    os.execute('mkdir -p '..save_path)  
    local record_file = io.open(save_path..'output.txt', 'w') 
    record_file:write('\n') 
    record_file:close()
    
    
    --Functions for training 
    local train_sample = function(mlp,criterion,x,y,learn_rate)
        local pred = mlp:forward(x)
        local err = criterion:forward(pred, y) 
        mlp:zeroGradParameters()
        local t = criterion:backward(pred, y)
        mlp:backward(x, t);
        mlp:updateParameters(learn_rate);
        return pred
    end
    
    local test_error = function(net,testData) 
    
        local testErrors = 0 
        
        for i = 1,testData:size() do 
            
            local sample,target = testData:next()  
            local output = net:forward(sample):float()  
            local _,pred = torch.max(output,2) 
            
            testErrors = testErrors + (pred:float()-target:float()):abs():gt(0):sum()  
              
        end 
        
        return testErrors 
    end 
   
    --===========Data========= 
    local trainData = nil 
    
    if trainData == nil then 
        trainData = torch.load('./Data/CIFAR/CIFAR_CN_train.t7')
        testData = torch.load('./Data/CIFAR/CIFAR_CN_test.t7')
        trainData.datacn = trainData.datacn:resize(50000,3,32,32):contiguous()  
        testData.datacn = testData.datacn:resize(10000,3,32,32):contiguous()  
        trainData.labels = trainData.labels:reshape(50000,1) 
        testData.labels = testData.labels:reshape(10000,1) 
        --data-params  
        nval = 1000 
        bsz = 128 
        valData = {} 
        valData.datacn = trainData.datacn:narrow(1,1,nval) 
        valData.labels = trainData.labels:narrow(1,1,nval)  
        trainData.datacn = trainData.datacn:narrow(1,nval+1,50000-nval)
        trainData.labels = trainData.labels:narrow(1,nval+1,50000-nval) 
        ds_train = DataSource({dataset = trainData.datacn, targets= trainData.labels, batchSize = bsz})
        ds_val = DataSource({dataset = valData.datacn, targets= valData.labels, batchSize = bsz})
        ds_test = DataSource({dataset = testData.datacn, targets= testData.labels, batchSize = bsz})
    end
    
    
        --=====architechture====
        --local 
        net = nn.Sequential()
        
        --layer1 
        local nfilt1 = 32 
        local pool1 = 'max'
        ----------------- 
        net:add(nn.SpatialZeroPadding(2, 2, 2, 2))
        if trained_layers[1] then 
            net:add(trained_layers[1]) 
        else 
            net:add(nn.SpatialConvolutionFFT(3, nfilt1, 5, 5))    
        end 
        net:add(nn.Threshold())
        if pool1 == 'max' then 
            net:add(nn.SpatialMaxPooling(2, 2))
        elseif pool1 == 'L2' then 
            net:add(nn.SpatialLPPooling(nfilt1, 2, 2, 2))
        end
        net.W1 = net:get(2) 
    
        --layer2 
        local nfilt2 = 128 
        local pool2 = 'offset'
        --local pool2 = 'max'
        local nOutputPlane2
        print('pool2: '..pool2) 
        if pool2 ~= 'offset' then 
            nfilt2 = nfilt2 * 3 
            nOutputPlane2 = nfilt2 
        elseif pool2 == 'offset' then  
            nOutputPlane2 = nfilt2 * 3 
        end
        ----------------- 
        net:add(nn.SpatialZeroPadding(2, 2, 2, 2))
        if trained_layers[2] then 
            net:add(trained_layers[2]) 
        else 
            net:add(nn.SpatialConvolutionFFT(nfilt1, nfilt2, 5, 5))
        end
        net:add(nn.Threshold())
        if pool2 == 'max' then 
            net:add(nn.SpatialMaxPooling(2, 2))
        elseif pool2 == 'L2' then 
            net:add(nn.SpatialLPPooling(nfilt2, 2, 2, 2))
        elseif pool2 == 'offset' then 
            net:add(offset_pooling(2, 2, 1))
        end
        net.W2 = net:get(6) 
    
        --layer3 
        local nfilt3 = 128
        local pool3 = 'offset'
        --local pool3 = 'max'
        print('pool3: '..pool3) 
        if pool3 ~= 'offset' then 
            nfilt3 = nfilt3 * 3 
            nOutputPlane3 = nfilt3 
        elseif pool3 == 'offset' then  
            nOutputPlane3 = nfilt3 * 3 
        end
        -----------------
        net:add(nn.SpatialZeroPadding(2, 2, 2, 2))
        if trained_layers[3] then 
            net:add(trained_layers[3]) 
        else 
            net:add(nn.SpatialConvolutionFFT(nOutputPlane2, nfilt3, 5, 5))
        end 
        net:add(nn.Threshold())
        if pool3 == 'max' then 
            net:add(nn.SpatialMaxPooling(2, 2))
        elseif pool3 == 'L2' then 
            net:add(nn.SpatialLPPooling(nfilt3, 2, 2, 2))
        elseif pool3 == 'offset' then 
            net:add(offset_pooling(2, 2, 1))
        end
        net.W3 = net:get(10) 
        
        --fully-connected layer 
        local size = nOutputPlane3 * 4 * 4
        net:add(nn.Reshape(size))
        net:add(nn.Linear(size, 10))
        net:add(nn.LogSoftMax())
        net:cuda() 
        
        --save diagnostic data   
        local I = image_hyper_spectral(net.W1.weight:float(),8,1)
        image.save(save_path..paths.basename(save_path)..'_layer1_init.png',I) 
        local I = image_hyper_spectral(net.W2.weight:float(),8,1)
        image.save(save_path..paths.basename(save_path)..'_layer2_init.png',I) 
        local sample1, target1 = ds_train:next()
        local I = image.toDisplayTensor({input=sample1,nrow=16,padding=1})
        image.save(save_path..'sample.png',I)    
        local record_file = io.open(save_path..'output.txt', 'a') 
        record_file:write(tostring(target1)..'\n') 
        record_file:close() 
              
        --criterion
        local criterion = nn.ClassNLLCriterion():cuda()
         
        local train_error_plot = torch.zeros(epochs) 
        local test_error_plot = torch.zeros(epochs) 
        local valid_error_plot = torch.zeros(epochs) 
       
        --Training 
        for i = 1,epochs do 
        
            if i == 10 then 
                learn_rate = learn_rate/5 
            end 

            sys.tic() 
            trainErrors = 0 
            
            for j = 1,ds_train:size() do
                
               progress(j,ds_train:size()) 
        
               sample,target = ds_train:next() 
               target = target:resize(bsz) 
               output = train_sample(net,criterion,sample,target,learn_rate):float()  
               _,pred = torch.max(output,2) 
               trainErrors = trainErrors + (pred:float()-target:float()):abs():gt(0):sum()  
        
            end 
            
            --Measure Test Error
            local valErrors = test_error(net,ds_val) 
            local testErrors = test_error(net,ds_test)
            
            local valid_percent_error = valErrors/ds_val.data[1]:size(1)*100 
            local train_percent_error = trainErrors/ds_train.data[1]:size(1)*100 
            local test_percent_error =  testErrors/ds_test.data[1]:size(1)*100
    
            local epoch_output = tostring(i)..'. TIME: '..tostring(sys.toc())..' Train Error: '..tostring(train_percent_error)..'%'..
             ' Val Error: '..tostring(valid_percent_error)..'%'..' Test Error: '..tostring(test_percent_error)..'%'  
    
            train_error_plot[i] = train_percent_error 
            test_error_plot[i] = test_percent_error
            valid_error_plot[i] = valid_percent_error 
    
            gnuplot.plot({'train', train_error_plot, '+'}, {'test', test_error_plot, '+'}, {'valid', valid_error_plot, '+'})  
            gnuplot.figprint(save_path..'error_history.eps')     
            gnuplot.closeall()
            
            print(epoch_output)
            local record_file = io.open(save_path..'output.txt', 'a') 
            record_file:write(epoch_output..'\n') 
            record_file:close() 
        
            local I = image_hyper_spectral(net.W1.weight:float(),8,1)
            image.save(save_path..paths.basename(save_path)..'_layer1.png',I) 
            local I = image_hyper_spectral(net.W2.weight:float(),8,1)
            image.save(save_path..paths.basename(save_path)..'_layer2.png',I) 
    
        end
        
end 

dofile('init.lua')
dofile('networks.lua') 
torch.manualSeed(1)
math.randomseed(1)
cutorch.setDevice(4) 

--evaluate first layer filters 
save_path = './Results/CIFAR/'
epochs = 100 
learn_rate = 0.05  
trained_layers = {} 

eval_on_CIFAR(trained_layers,learn_rate,epochs,save_path) 


