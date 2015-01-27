if nn.MulConstant == nil then 
    dofile('./Modules/MulConstant.lua') 
end 

offset_pooling = function(kW,kH,phase_const,mod_const,zero_offset_gradInput) 

    local phase_const = phase_const or 1 
    local mod_const = mod_const or 1 
    local modulus = nn.Sequential()
    modulus:add(nn.MulConstant(mod_const))
    modulus:add(nn.SpatialMaxPooling(kW,kH,kW,kH))

    local phase = nn.Sequential() 
    phase:add(nn.MulConstant(phase_const)) 
    phase:add(jzt.SSMPoolingOffsets(kW,kH,zero_offset_gradInput))
    
    local pooling = nn.Sequential() 
    local m = nn.ConcatTable() 
    m:add(modulus) 
    m:add(phase) 
    pooling:add(m) 
    pooling:add(nn.JoinTable(2))
    
    return pooling

end 

--traditional convnet with max pooling 
conv_net2 = function(inputSize)  

    local net = nn.Sequential()
    local size = inputSize

    --1st Spatial Stage
    local nOutputPlane1 = 4 * 3 
    local k1 = 9
    local pad1 = (k1-1)/2 
    if (nn.SpatialConvolutionMM == nil) then
      net:add(nn.SpatialPadding(pad1,pad1,pad1,pad1,3,4))
      net:add(nn.SpatialConvolutionFFT(inputSize[1],nOutputPlane1,k1,k1))
    else
      net:add(nn.SpatialConvolutionMM(inputSize[1],nOutputPlane1,k1,k1, pad1))
    end 
    net:add(nn.SpatialMaxPooling(pool1,pool1))
    net:add(nn.Threshold(0,0))
    size[1] = nOutputPlane1 
    size[2] = math.floor((size[2] - pool1)/pool1 + 1) 
    size[3] = math.floor((size[3] - pool1)/pool1 + 1) 
    --2nd Spatial Stage 
    local nInputPlane2 = nOutputPlane1
    local nOutputPlane2 = 1 * 3
    local k2 = 9
    local pad2 = (k2-1)/2 
    if (nn.SpatialConvolutionMM == nil) then
      net:add(nn.SpatialPadding(pad2,pad2,pad2,pad2,3,4))
      net:add(nn.SpatialConvolutionFFT(nInputPlane2,nOutputPlane2,k2,k2))
    else
      net:add(nn.SpatialConvolutionMM(nInputPlane2,nOutputPlane2,k2,k2, pad2))
    end
    net:add(nn.SpatialMaxPooling(pool2,pool2))
    net:add(nn.Threshold(0,0))
    size[1] = nOutputPlane2 
    size[2] = math.floor((size[2] - pool2)/pool2 + 1) 
    size[3] = math.floor((size[3] - pool2)/pool2 + 1) 
    --Reshape for F.C. stages 
    size = size[1]*size[2]*size[3]
    net:add(nn.Reshape(size))
    net:add(nn.Linear(size,2))
    net:add(nn.Threshold(0,0)) 

    return net 

end

--convnet with max offset pooling 
phase_net1 = function(inputSize,pool1,phase_const,mag_const,flag)  

    local net = nn.Sequential()
    local size = inputSize
    local zero_offset_gradInput = true 

    --1st Spatial Stage
    local nOutputPlane1 = 1  
    local k1 = 9
    local pad1 = (k1-1)/2 
    net:add(nn.SpatialPadding(pad1,pad1,pad1,pad1,3,4))
    net:add(nn.SpatialConvolutionFFT(inputSize[1],nOutputPlane1,k1,k1))
    net:add(nn.Threshold(0,0))
    --pooling1 = offset_pooling(pool1,pool1,phase_const,mag_const,zero_offset_gradInput)
    pooling1 = nn.SpatialMaxPooling(pool1,pool1,pool1,pool1) 
    net:add(pooling1)
    net.pooling1 = pooling1 
    size[1] = 1 * nOutputPlane1  -- TODO Jake.. what is 3.. There could be incorrect
    size[2] = math.floor((size[2] - pool1)/pool1 + 1) 
    size[3] = math.floor((size[3] - pool1)/pool1 + 1) 
    --Reshape for F.C. stages 
    size = size[1]*size[2]*size[3]
    net:add(nn.Reshape(size))
    if flag == nil then
       net:add(nn.Linear(size,2)) 
    else
       net:add(nn.Linear(size,4)) 
    end

    return net 

end
--convnet with max offset pooling 
phase_net2 = function(inputSize,pool1,pool2,phase_const,mag_const,flag)  

    local net = nn.Sequential()
    local size = inputSize
    local zero_offset_gradInput = true 

    --1st Spatial Stage
    local nOutputPlane1 = 4  
    local k1 = 9
    local pad1 = (k1-1)/2 
    net:add(nn.SpatialPadding(pad1,pad1,pad1,pad1,3,4))
    net:add(nn.SpatialConvolutionFFT(inputSize[1],nOutputPlane1,k1,k1))
    net:add(nn.Threshold(0,0))
    pooling1 = offset_pooling(pool1,pool1,phase_const,mag_const,zero_offset_gradInput)
    net:add(pooling1)
    net.pooling1 = pooling1 
    size[1] = 3 * nOutputPlane1 
    size[2] = math.floor((size[2] - pool1)/pool1 + 1) 
    size[3] = math.floor((size[3] - pool1)/pool1 + 1) 
    --2nd Spatial Stage 
    local nInputPlane2 = 3 * nOutputPlane1
    local nOutputPlane2 = 1 
    local k2 = 9
    local pad2 = (k2-1)/2 
    net:add(nn.SpatialPadding(pad2,pad2,pad2,pad2,3,4))
    net:add(nn.SpatialConvolutionFFT(nInputPlane2,nOutputPlane2,k2,k2))
    pooling2 = offset_pooling(pool2,pool2,phase_const,mag_const,zero_offset_gradInput)
    net:add(pooling2)
    net.pooling2 = pooling2 
    net:add(nn.Threshold(0,0))
    size[1] = 3 * nOutputPlane2 
    size[2] = math.floor((size[2] - pool2)/pool2 + 1) 
    size[3] = math.floor((size[3] - pool2)/pool2 + 1) 
    --Reshape for F.C. stages 
    size = size[1]*size[2]*size[3]
    net:add(nn.Reshape(size))
    if flag == nil then
       net:add(nn.Linear(size,2)) 
    else
       net:add(nn.Linear(size,4)) 
    end

    return net 

end
