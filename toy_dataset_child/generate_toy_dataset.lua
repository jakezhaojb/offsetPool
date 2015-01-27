torch.setdefaulttensortype('torch.FloatTensor')
if math.mod == nil then
  math.mod = math.fmod
end

require('image')
dofile('../jitter_utils.lua')

im = image.load('./test_image.png')
im = image.scale(im,200,200)
im = image.rgb2y(im:narrow(1,1,3)) 
im = padImage(im, {1, 1, 1, 1}, 0)  -- image, pad_lrtb, bordervalue
w = im:size(3)
h = im:size(2)

uv = torch.FloatTensor({w/2,h/2})
--h1 = image.display{image=im[{{1,3},{},{}}], zoom = 5}

-- Draw random stewies on top of each other
out_im = im:clone():fill(0)
--math.randomseed(0)
num_images = 3000
dataset = {} 
dataset.images = torch.Tensor(num_images,1,h,w) 

uv = torch.FloatTensor(2, num_images)
uv[{1,{}}]:fill(w/2)
uv[{2,{}}]:fill(h/2)

scale = 0.5

for i = 1, num_images do
  
  deg_rot = math.random() * 360 
  trans_u = (math.random() - 0.5) * w
  trans_v = (math.random() - 0.5) * h

  rand_im = distortImage{im=im, deg_rot=deg_rot, scale=scale, trans_u_pix=trans_u, trans_v_pix = trans_v}
  
  dataset.images[i]:copy(rand_im)

  distortPos2D(uv[{{},{i}}], deg_rot, scale, trans_v, trans_u, w, h)
end

dataset.images:add(-dataset.images:mean())

dataset.targets = uv:t()

--torch.save('./Data/toy_data/stewie_valid.t7',dataset) 
torch.save('./toy.t7',dataset) 

--for i = 1, num_images do
-- print(i)  
--  uv = dataset.targets[i] 
--  dataset.images[i][1][math.floor(uv[2])][math.floor(uv[1])] = 0 
--  --drawLabels(dataset.images[i], dataset.targets[i], w, h, 1)
--end
--
--h2 = image.display{image=dataset.images, zoom = 1, padding = 1}
