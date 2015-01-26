bsz = 1
inplane = 3
W = 11
H = 11
kW = 2
kH = 2 
input = torch.ones(1,W*H):reshape(1,1,H,W):repeatTensor(bsz,inplane,1,1) 
input:div(input:max()) 

--module 
self = {} 
self.gridX = torch.linspace(-1,1,kW)
self.gridY = torch.linspace(-1,1,kH)
--fprop 
inputSize = input:size() 
nOutputPlane = 2*inputSize[2] 
nOutputCols = math.floor(inputSize[4]/kW)
nOutputRows = math.floor(inputSize[3]/kH) 
maxW = nOutputCols*kW
maxH = nOutputRows*kH 
self.output = self.output or torch.Tensor(inputSize[1],nOutputPlane,nOutputRows,nOutputCols):typeAs(input)
self.softmax = self.softmax or torch.Tensor():resize(bsz,inputSize[2],maxW,maxH):typeAs(input) 

for batch = 1,inputSize[1] do 
    for inplane = 1,inputSize[2] do 
        local oi = 1 
        for i = 1,inputSize[3]-1,kH do 
            local oj = 1 
            for j = 1,inputSize[4]-1,kW do 
               
                local pool_sum = 0 
                local dx = 0
                local dy = 0 

                for h = 0,kH-1 do 
                    for w = 0,kW-1 do
                        pool_sum = pool_sum + math.exp(input[batch][inplane][i+h][j+w])                      
                    end
                end
               
                for h = 0,kH-1 do 
                    for w = 0,kW-1 do 
                        local val = math.exp(input[batch][inplane][i+h][j+w])/pool_sum                      
                        self.softmax[batch][inplane][i+h][j+w] = val                    
                        dx = dx + self.gridX[w+1]*val 
                        dy = dy + self.gridY[h+1]*val
                    end
                end
               
                self.output[batch][2*inplane-1][oi][oj] = dx 
                self.output[batch][2*inplane][oi][oj] = dy 
                
                oj = oj + 1 
            end
            oi = oi + 1 
        end
    end
end

--bprop
gradOutput = torch.ones(self.output:size())
self.gradInput = torch.Tensor():resizeAs(input):fill(0)  

for batch = 1,inputSize[1] do 
    for inplane = 1,inputSize[2] do 
        local oi = 1 
        for i = 1,inputSize[3]-1,kH do 
            local oj = 1 
            for j = 1,inputSize[4]-1,kW do 
               
                local pool_sum_X = 0  
                local pool_sum_Y = 0  
                
                for h = 0,kH-1 do 
                    for w = 0,kW-1 do
                        pool_sum_X = pool_sum_X + self.softmax[batch][inplane][i+h][i+w]*self.gridX[w+1] 
                        pool_sum_Y = pool_sum_Y + self.softmax[batch][inplane][i+h][i+w]*self.gridY[h+1] 
                    end
                end

                for h = 0,kH-1 do 
                    for w = 0,kW-1 do 
                        
                        local softmax_pool = self.softmax[batch][inplane][i+h][j+w] 
                        local gradOutput_X = gradOutput[batch][2*inplane-1][oi][oj]  
                        local gradOutput_Y = gradOutput[batch][2*inplane][oi][oj]  
                        local gradInput = softmax_pool * (gradOutput_X * (self.gridX[w+1] - pool_sum_X) + 
                                                          gradOutput_Y * (self.gridY[h+1] - pool_sum_Y))   
                        self.gradInput[batch][inplane][i+h][j+w] = gradInput 
                    end
                end
               
                oj = oj + 1 
            end
            oi = oi + 1 
        end
    end
end




