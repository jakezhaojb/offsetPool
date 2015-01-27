torch.setdefaulttensortype('torch.FloatTensor')
if math.mod == nil then
  math.mod = math.fmod
end

require('image')
dofile('../jitter_utils.lua')

im1 = image.load('test1.png')
im2 = image.load('test2.png')
im1 = image.scale(im1,32,32)
im1 = padImage(im1, {1, 1, 1, 1}, 0)  -- image, pad_lrtb, bordervalue
w = im1:size(3)
h = im1:size(2)
im2 = image.scale(im2,32,32)
im2 = padImage(im2, {1, 1, 1, 1}, 0)  -- image, pad_lrtb, bordervalue
w_ = im2:size(3)
h_ = im2:size(2)

assert(w == w_)
assert(h == h_)

local num_images = 1000
local scale = 0.5
-- im1
local uv = torch.FloatTensor({w/2,h/2})
--h1 = image.display{image=im[{{1,3},{},{}}], zoom = 5}

-- Draw random stewies on top of each other
local out_im = im1:clone():fill(0)
--math.randomseed(0)
local dataset1 = {} 
dataset1.images = torch.Tensor(num_images,1,h,w) 

uv = torch.FloatTensor(2, num_images)
uv[{1,{}}]:fill(w/2)
uv[{2,{}}]:fill(h/2)

for i = 1, num_images do
  
  local deg_rot = math.random() * 360 
  local trans_u = (math.random() - 0.5) * w
  local trans_v = (math.random() - 0.5) * h

  local rand_im = distortImage{im=im1, deg_rot=deg_rot, scale=scale, trans_u_pix=trans_u, trans_v_pix = trans_v}
  
  dataset1.images[i]:copy(rand_im)

  distortPos2D(uv[{{},{i}}], deg_rot, scale, trans_v, trans_u, w, h)
end

dataset1.targets = uv:t()

-- im2
uv = torch.FloatTensor({w/2,h/2})
--h1 = image.display{image=im[{{1,3},{},{}}], zoom = 5}

-- Draw random stewies on top of each other
out_im = im2:clone():fill(0)
--math.randomseed(0)
local dataset2 = {} 
dataset2.images = torch.Tensor(num_images,1,h,w) 

uv = torch.FloatTensor(2, num_images)
uv[{1,{}}]:fill(w/2)
uv[{2,{}}]:fill(h/2)

for i = 1, num_images do
  
  local deg_rot = math.random() * 360 
  local trans_u = (math.random() - 0.5) * w
  local trans_v = (math.random() - 0.5) * h

  local rand_im = distortImage{im=im1, deg_rot=deg_rot, scale=scale, trans_u_pix=trans_u, trans_v_pix = trans_v}
  
  dataset2.images[i]:copy(rand_im)

  distortPos2D(uv[{{},{i}}], deg_rot, scale, trans_v, trans_u, w, h)
end

dataset2.targets = uv:t()

-- merge two datasets
dataset = {}
dataset.targets = torch.cat(dataset1.targets, dataset2.targets, 2)
dataset.images = dataset1.images + dataset2.images

dataset.images:add(-dataset.images:mean())

torch.save('./toy.t7',dataset) 

--torch.save('./Data/toy_data/stewie_valid.t7',dataset) 

--for i = 1, num_images do
-- print(i)  
--  uv = dataset.targets[i] 
--  dataset.images[i][1][math.floor(uv[2])][math.floor(uv[1])] = 0 
--  --drawLabels(dataset.images[i], dataset.targets[i], w, h, 1)
--end
--
--h2 = image.display{image=dataset.images, zoom = 1, padding = 1}
