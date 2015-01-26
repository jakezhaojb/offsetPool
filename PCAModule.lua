local PCA, parent = torch.class('nn.PCA', 'nn.Module')

function PCA:compute_pca(data,nbasis)
    
    local dims = data:size()
    local nsamples = dims[1]
    local n_dimensions 
    if data:dim() == 4 then 
        n_dimensions = dims[2] * dims[3] * dims[4]
    elseif data:dim() == 2 then 
        n_dimensions = dims[2] 
    else 
        error('PCA module: init data must be 4D or 2D tensor') 
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

function PCA:__init(data, outputSize)
   parent.__init(self)
   --if outputSize == nil then the module will 
   --keep the top 10% highest energy components 
   local ce,cv,mean
   if data ~= nil then 
      ce,cv,mean = self:compute_pca(data, outputSize)   
   else 
      error('PCA Module: [data] is nil') 
   end
   self.basis = cv:t()  
   self.mean = mean
   self.inputCentered = torch.Tensor() 
   self.inputSize = self.basis:size(2) 
   self.outputSize = self.basis:size(1) 
end

function PCA:reset(data, outputSize)
   local ce,cv,mean = self:compute_pca(data, outputSize)   
   self.basis = cv:t()  
   self.mean=mean 
end

function PCA:updateOutput(input)
   
   self.inputCentered:resizeAs(input) 
   
   if input:dim() == 1 then
      
      if input:size(1)~=self.inputSize then 
         error('PCA Module: [input]-dim does not match [data]-dim')
      end
      self.output:resize(self.outputSize)
      self.inputCentered:copy(input):add(-self.mean) 
      self.output:addmv(1, self.basis, self.inputCentered)
   
   elseif input:dim() == 2 then
      
      if input:size(2)~=self.inputSize then 
         error('PCA Module: [input]-dim does not match [data]-dim')
      end
      
      local bsz = input:size(1)
      self.output:resize(bsz,self.outputSize)
      
      --will throw error if [bsz] is changed while training 
      if self.mean:size(1)~=bsz then 
         self.mean = self.mean:repeatTensor(bsz,1)
      end
      
      self.inputCentered:copy(input):add(-self.mean) 
      self.output:addmm(1, self.inputCentered, self.basis)
   
   else
      error('input must be vector or matrix')
   end

   return self.output
end

