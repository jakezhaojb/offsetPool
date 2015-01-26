local SpatialSoftMax, parent = torch.class('jzt.SpatialSoftMax', 'nn.Module')

function SpatialSoftMax:__init(kW, kH)
   parent.__init(self)

   self.kW = kW
   self.kH = kH

end

function SpatialSoftMax:updateOutput(input)
   jzt.SpatialSoftMax_updateOutput(self, input)
   return self.output
end

function SpatialSoftMax:updateGradInput(input, gradOutput)
   jzt.SpatialSoftMax_updateGradInput(self, input, gradOutput)
   return self.gradInput
end

function SpatialSoftMax:empty()
   self.gradInput:resize()
   self.gradInput:storage():resize(0)
   self.output:resize()
   self.output:storage():resize(0)
end
