torch.setdefaulttensortype('torch.FloatTensor')
if math.mod == nil then
  math.mod = math.fmod
end

require('image')
dofile('jitter_utils.lua')

im = image.load('test_image.png')
im = padImage(im, {1, 1, 1, 1}, 0)  -- image, pad_lrtb, bordervalue
w = im:size(3)
h = im:size(2)

uv = torch.FloatTensor({w/2,h/2})
drawLabels(im, uv, w, h, 0, 0.02)
h1 = image.display{image=im[{{1,3},{},{}}], zoom = 5}

-- Draw random stewies on top of each other
out_im = im:clone():fill(0)
--math.randomseed(0)
rand_im = {}
num_images = 1
uv = torch.FloatTensor(2, num_images)
uv[{1,{}}]:fill(w/2)
uv[{2,{}}]:fill(h/2)

for i = 1, num_images do
  
 -- deg_rot = math.random() * 360
 -- scale = math.random()*0.6  + 0.4
 -- trans_u = (math.random() - 0.5) * w
 -- trans_v = (math.random() - 0.5) * h
  
  deg_rot = 0
  scale = 0.5
  trans_u = (math.random() - 0.5) * w
  trans_v = (math.random() - 0.5) * h

  rand_im, flow = distortImage{im=im, deg_rot=deg_rot, scale=scale, 
  trans_u_pix=trans_u, trans_v_pix = trans_v}
  out_im = alphaBlend(rand_im, out_im)  -- rand_im on top of out_im
  distortPos2D(uv[{{},{i}}], deg_rot, scale, trans_v, trans_u, w, h)
end

for i = 1, num_images do
  drawLabels(out_im, uv, w, h, 1)
end

h2 = image.display{image=out_im[{{1,3},{},{}}], zoom = 5}
