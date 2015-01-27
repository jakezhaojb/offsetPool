--require 'mattorch'
require 'cunn'
require 'FFTconv'
require 'image'
require 'xlua'
require 'optim'
require 'nnx' 
require 'unsup' 
require 'data' 
require 'gnuplot' 

local hasjzt,jzt = pcall(require, "jzt")
if not hasjzt then
  print('Couldnt find jzt, continuing anyway')
end

dofile('util.lua') 
dofile('pbar.lua')
dofile('Test.lua')
dofile('jitter_utils.lua')
dofile('networks.lua') 
