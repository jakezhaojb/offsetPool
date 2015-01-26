local SpatialSoftMax, parent = torch.class('nn.SpatialSoftMax', 'nn.Module')
function SpatialSoftMax:__init(kW, kH, beta) 
    
    parent.__init(self) 
    self.kW = kW 
    self.kH = kH 
    self.beta = beta or 1
    self.output = torch.Tensor() 
    self.norm = torch.Tensor() 
    self.gradInput = torch.Tensor()  

end

function SpatialSoftMax:updateOutput(input) 
    
    local kW = self.kW 
    local kH = self.kH
    local bsz = input:size(1) 
    local inplane = input:size(2) 
    local inW = input:size(3) 
    local inH = input:size(4)
    local outW = math.floor(inW/kW) 
    local outH = math.floor(inH/kH) 

    self.output:typeAs(input):resizeAs(input) 
    self.norm:typeAs(input):resize(bsz*inplane,inH,inW) 
    self.output:copy(input) 
    self.output:mul(self.beta):exp() 
    
    self.norm:copy(self.output:resize(bsz*inplane,inH,inW)) 
    self.norm = self.norm:unfold(2,kH,kH):unfold(3,kW,kW):contiguous():resize(bsz*inplane,outy,outx,py*px)
    self.norm = self.norm:sum(4):resize(bsz*inplane,outH,outW,1,1) 
    self.norm = self.norm:expand(bsz*inplane,outH,outW,kH,kW):transpose(3,4):contiguous():resize(bsz*inplane,inH,inW)
    self.output:cdiv(self.norm)

    return self.output 
end

function SpatialSoftMax:updateGradInput(input, gradOutput) 
    --input and gradOutput are of the same size 
    self.gradInput = self.gradInput:typeAs(input):resizeAs(input):copy(gradOutput)  
    self.gradInput:cmul(self.maxIdx)   
    
    return self.gradInput 
end  

--these are empty because there are no 
--trainable paramters in this module 
function SpatialSoftMax:zeroGradParameters()

end

function SpatialSoftMax:accGradParameters()

end

function SpatialSoftMax:updateParameters()

end
