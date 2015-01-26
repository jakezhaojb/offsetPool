dofile('init.lua') 
if nn.PCA == nil then 
    dofile('PCAModule.lua') 
end

pca = function(data, nbasis)

    local dims = data:size()
    local nsamples = dims[1]
    local n_dimensions 
    if data:dim() == 4 then 
        n_dimensions = dims[2] * dims[3] * dims[4]
    elseif data:dim() == 2 then 
        n_dimensions = dims[2] 
    else 
        error('zca_whiten: data must be 4D or 2D tensor') 
    end 

    local mdata = data:reshape(nsamples, n_dimensions)
    local mean = torch.mean(mdata,1):squeeze() 
    mdata:add(torch.ger(torch.ones(nsamples),mean):mul(-1))
    
    local ce, cv = unsup.pcacov(data:reshape(nsamples, n_dimensions))
    
    if nbasis == nil then 
        --keep only to 10% of the componenets 
        local cutoff = cutoff or 0.10 
        local csum = torch.cumsum(ce,1)
        local thresh = csum[csum:size(1)]*cutoff
        local tmp = csum:clone():add(-thresh):abs()
        local _,idx = torch.min(tmp,1)
        idx = idx[1] 
        ce = ce:narrow(1,idx,ce:size(1)-idx+1)
        cv = cv:narrow(1,idx,cv:size(1)-idx+1)
    else 
        ce = ce:narrow(1,ce:size(1)-nbasis+1,nbasis)
        cv = cv:narrow(1,cv:size(1)-nbasis+1,nbasis)
    end

    return ce,cv,mean 

end


x ={-0.1299,   
    0.6030,   
    2.1742,  
    2.9014,   
    0.9621,   
    1.6007,   
    1.4110,   
   -0.5694,   
    1.1785,   
    0.1349}   

y = {-2.0586,
    -1.4216,
     0.1097,
    -0.1230,
    -1.3925,
    -0.4827,
    -0.6454,
    -1.8057,
    -0.7251,
    -1.4839}

X = torch.Tensor({y,x}):t()

bsz = 16 
m = nn.PCA(X,2) 
sample = torch.randn(bsz,2)
 
out = m:forward(sample) 


--
--ce,cv,mean = pca(X,outdim)
--
----invert 
--sample = torch.ones(bsz,2):div(math.sqrt(2)) 
--sample_cent = sample:clone():add(-mean:repeatTensor(bsz,1))
--coeff = torch.mm(sample_cent,cv)



