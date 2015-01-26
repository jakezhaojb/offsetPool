require("image")

local function load_image(dir) 

    local im = image.load(dir)  
    -- Check channels
    if im:dim() == 2 then
       local new_im = torch.Tensor(3,im:size(1),im:size(2))
       for c = 1,3 do
      new_im:select(1,c):copy(im)
       end
       im = new_im
    elseif im:size(1) == 1 then
       local new_im = torch.Tensor(3,im:size(2),im:size(3))
       for c = 1,3 do
      new_im:select(1,c):copy(im:select(1,1))
       end
       im = new_im
    end
    if im:dim() ~= 3 or im:size(1) ~=3 then
       error("Image channel is not 3")
    end
    return im 

end

--[[
-- distort function
function distort(i, deg_rot, scale, trans_v, trans_u, xsheer, ysheer)
    -- size:
    local height,width = i:size(2),i:size(3)
    -- x/y grids
    local grid_y = torch.ger( torch.linspace(-1,1,height), torch.ones(width) )
    local grid_x = torch.ger( torch.ones(height), torch.linspace(-1,1,width) )
    local flow = torch.FloatTensor()
    local flow_scale = torch.FloatTensor()
    local flow_rot = torch.FloatTensor()
    -- global flow:
    flow:resize(2,height,width)
    flow:zero()
    local rot_angle
    local rotmat

    -- Apply translation (comes before rotation)
    flow[1]:add(trans_v * height)
    flow[2]:add(trans_u * width)
    -- Apply scale and rotation
    flow_rot:resize(2,height,width)
    flow_rot[1] = grid_y * ((height-1)/2) * -1
    flow_rot[2] = grid_x * ((width-1)/2) * -1
    local view = flow_rot:reshape(2,height*width)
    
    local function rmat(deg, s)
        local r = deg/180*math.pi
        return torch.FloatTensor{{s * math.cos(r), -s * math.sin(r)},
                                {s * math.sin(r), s * math.cos(r)}}
    end
    
    local function smat(xsheer, ysheer)
        return torch.FloatTensor{{1, xsheer},
                                 {ysheer,1}}
    end
    
    rotmat = rmat(deg_rot, 1 + scale)
    shemat = smat(xsheer, ysheer) 
    flow_sheerr = torch.mm(shemat, view) 
    flow_rotr = torch.mm(rotmat, view)
    flow_sheer = flow_rot - flow_sheerr:reshape(2, height, width) 
    flow_rot = flow_rot - flow_rotr:reshape(2, height, width)
    flow:add(flow_rot)
    flow:add(flow_sheer) 
    -- apply field
    local result = torch.FloatTensor()
    image.warp(result,i,flow,'lanczos')
    return result, rotmat
end
--]]

-- deg_rot (just degrees of rotation about center)
-- trans_v_pix and trans_u_pix are the translation size in pixels
-- scale is >0 scale factor (default = 1)
-- TODO: This function will create many memory allocations!
function distortImage(...)
  local _, im, deg_rot, scale, trans_v_pix, trans_u_pix = dok.unpack(
    {...},
    'ScanWindowSampler.distortImage',
    'distort a single image',
    {arg='im', type='torch.Tensor', help='image (KxHxW)', req=true},
    {arg='deg_rot', type='number', help='degrees rotation', default=0},
    {arg='scale', type='number', help='image scaling', default=1},
    {arg='trans_v_pix', type='number', help='v translation', default=0},
    {arg='trans_u_pix', type='number', help='u translation', default=0}
  )
  assert(im:dim() == 3, 'Input image is not 3 dimensional!')
  assert(torch.typename(im) == 'torch.FloatTensor', 'Input not FloatTensor!')

  -- size:
  local height, width = im:size(2), im:size(3)

  -- x/y grids
  local grid_y = torch.ger(torch.linspace(-1,1,height), torch.ones(width))
  local grid_x = torch.ger(torch.ones(height), torch.linspace(-1,1,width))

  local flow = torch.FloatTensor()
  local flow_scale = torch.FloatTensor()
  local flow_rot = torch.FloatTensor()

  -- global flow:
  flow:resize(2,height,width)
  flow:zero()

  -- Apply translation (comes before rotation)
  flow[1]:add(trans_v_pix)
  flow[2]:add(trans_u_pix)

  -- Apply scale and rotation
  flow_rot:resize(2,height,width)
  flow_rot[1] = grid_y * ((height-1)/2) * -1
  flow_rot[2] = grid_x * ((width-1)/2) * -1

  local view = flow_rot:reshape(2,height*width)
  local rotmat = torch.FloatTensor(2,2)  -- TODO: Preallocate this
  rmat(rotmat, deg_rot, 1/scale)

  local flow_rotr = torch.mm(rotmat, view)
  flow_rot = flow_rot - flow_rotr:reshape( 2, height, width )
  flow:add(flow_rot)

  local im_warp = im:clone():fill(0)

  -- apply field
  image.warp(im_warp, im, flow, 'bilinear')

  return im_warp
end

function crop(im, height, width, method, cropped) 
    local cropped = cropped or torch.Tensor(3,height,width)
    local width,height = cropped:size(3),cropped:size(2)
    local cstartx,cstarty,cendx,cendy = 1,1,width,height
    local startx, starty,endx,endy
    -- Determine start and end indices based on type of crop
    if method == "center" or method == nil then
        startx = math.modf((im:size(3) - width)/2) + 1
        starty = math.modf((im:size(2) - height)/2) + 1
    elseif method == "random" then
        startx = math.random(math.min(1,im:size(3) - width + 1), math.max(1,im:size(3) - width + 1))
        starty = math.random(math.min(1,im:size(2) - height + 1), math.max(1,im:size(2) - height + 1))
    elseif method == "leftupper" then
        startx = 1
        starty = 1
    elseif method == "leftlower" then
        startx = 1
        starty = im:size(2) - height + 1
    elseif method == "rightupper" then
        startx = im:size(3) - width + 1
        starty = 1
    elseif method == "rightlower" then
        startx = im:size(3) - width + 1
        starty = im:size(2) - height + 1
    elseif method == "corners" then
        local method = math.random(5)
        if method == 1 then -- Center
           startx = math.modf((im:size(3) - width)/2) + 1
           starty = math.modf((im:size(2) - height)/2) + 1
        elseif method == 2 then -- LeftUpper
           startx = 1
           starty = 1
        elseif method == 3 then -- LeftLower
           startx = 1
           starty = im:size(2) - height + 1
        elseif method == 4 then -- RightUpper
           startx = im:size(3) - width + 1
           starty = 1
        elseif method == 5 then -- RightLower
           startx = im:size(3) - width + 1
           starty = im:size(2) - height + 1
        end
    else
        error("Unrecognized cropping method")
    end
    endx = startx + width - 1
    endy = starty + height - 1
    -- Centering the image patch
    cstartx = startx
    cstarty = starty
    -- Rectify the indices for image
    startx = math.max(startx,1)
    endx = math.min(endx,im:size(3))
    starty = math.max(starty,1)
    endy = math.min(endy,im:size(2))
    -- Rectify end indices for cropped
    cstartx = startx - cstartx + 1
    cstarty = starty - cstarty + 1
    cendx = cstartx + endx - startx
    cendy = cstarty + endy - starty
    cropped:fill(0)
    cropped[{{},{cstarty,cendy},{cstartx,cendx}}]:copy(im[{{},{starty,endy},{startx,endx}}])
    return cropped 
end

-- addBorder: Add a border to the image, where the pixel values are clamped 
function addBorder(im, border_size)
  assert(im:dim() == 2, 'input must be dimension 2')
  local ret_sz = {im:size(1)+2*border_size, im:size(2)+2*border_size}
  local ret_im = im:clone():resize(unpack(ret_sz))
  
  local vbl = {1, border_size}  -- v range border left
  local v = {vbl[2] + 1,  vbl[2] + 1 + im:size(im:dim()-1) - 1}  -- v range img
  local vbr = {v[2] + 1, v[2] + 1 + border_size - 1}  -- v range border right
  
  local ubl = {1, border_size}
  local u = {ubl[2] + 1, ubl[2] + 1 + im:size(im:dim()) - 1}
  local ubr = {u[2] + 1, u[2] + 1 + border_size - 1}
  
  -- center
  ret_im[{v, u}]:copy(im)
  -- up
  ret_im[{vbl, u}]:copy(im[{{1},{}}]:expandAs(ret_im[{vbl, u}]))
  -- down
  ret_im[{vbr, u}]:copy(im[{{im:size(1)},{}}]:expandAs(ret_im[{vbr, u}]))
  -- left
  ret_im[{v, ubl}]:copy(im[{{},{1}}]:expandAs(ret_im[{v, ubl}]))
  -- right
  ret_im[{v, ubr}]:copy(im[{{},{im:size(2)}}]:expandAs(ret_im[{v, ubr}]))
  -- left + up
  ret_im[{vbl, ubl}] = im[1][1]
  -- right + up
  ret_im[{vbl, ubr}] = im[1][im:size(2)]
  -- left + down
  ret_im[{vbr, ubl}] = im[im:size(1)][1]
  -- right + down
  ret_im[{vbr, ubr}] = im[im:size(1)][im:size(2)]
  
  return ret_im
end

-- addBorder: Add a border to the right of the image, where the pixel values are
-- clamped 
function addBorderBottomAndRight(im, border_size)
  assert(im:dim() == 2, 'input must be dimension 2')
  local ret_sz = {im:size(1)+border_size, im:size(2)+border_size}
  local ret_im = im:clone():resize(unpack(ret_sz))
  
  local v = {1,  im:size(im:dim()-1)}  -- v range img
  local vbr = {v[2] + 1, v[2] + 1 + border_size - 1}  -- v range border right
  
  local u = {1, im:size(im:dim())}
  local ubr = {u[2] + 1, u[2] + 1 + border_size - 1}
  
  -- center
  ret_im[{v, u}]:copy(im)
  -- down
  ret_im[{vbr, u}]:copy(im[{{im:size(1)},{}}]:expandAs(ret_im[{vbr, u}]))
  -- right
  ret_im[{v, ubr}]:copy(im[{{},{im:size(2)}}]:expandAs(ret_im[{v, ubr}]))
  -- right + down
  ret_im[{vbr, ubr}] = im[im:size(1)][im:size(2)]
  
  return ret_im
  
end

function addConstantBorder(im, border_size, border_value)
  assert(im:dim() == 2, 'input must be dimension 2')
  local ret_sz = {im:size(1)+2*border_size, im:size(2)+2*border_size}
  local ret_im = im:clone():resize(unpack(ret_sz))
  
  local v = {border_size + 1,  border_size + 1 + im:size(im:dim()-1) - 1}  -- v range img
  local u = {border_size + 1, border_size + 1 + im:size(im:dim()) - 1}
  
  -- center
  ret_im:fill(border_value)
  ret_im[{v, u}]:copy(im)
  
  return ret_im
end

function padImage(im, pad_lrtb, border_value)
  assert(im:dim() == 3 or im:dim() == 2, 'input must be dimension 2 or 3')
  assert(#pad_lrtb == 4, 'Need 4 pad values')
  for i=1,#pad_lrtb do assert(pad_lrtb[i]>=0, 'Padding must be >= 0') end
  local ret_sz
  if im:dim() == 2 then
    ret_sz = {im:size(1)+pad_lrtb[1]+pad_lrtb[2], 
      im:size(2)+pad_lrtb[3]+pad_lrtb[4]}
  else
    ret_sz = {im:size(1), im:size(2)+pad_lrtb[3]+pad_lrtb[4], 
      im:size(3)+pad_lrtb[1]+pad_lrtb[2]}
  end
  local ret_im = im:clone():resize(unpack(ret_sz))
  
  local v = {pad_lrtb[3] + 1, pad_lrtb[3] + 1 + im:size(im:dim()-1) - 1}  -- v range img
  local u = {pad_lrtb[1] + 1, pad_lrtb[1] + 1 + im:size(im:dim()) - 1}
  
  -- center
  ret_im:fill(border_value)
  if im:dim() == 2 then
    ret_im[{v, u}]:copy(im)
  else
    ret_im[{{},v, u}]:copy(im)
  end
  
  return ret_im
end

-- Note: imA and imB will be modified!
-- This performs per-pixel alpha blending from here:
-- http://en.wikipedia.org/wiki/Alpha_compositing
function alphaBlend(imA, imB)
  local ret = imA:clone()
  assert(imA:size(1) == 4 and imB:size(1) == 4, 'You need an alpha chan')
  assert(imA:dim() == 3 and imB:dim() == 3, 'You need a 3D tensor')
  local tmp = torch.Tensor():typeAs(imA):resize(1, imA:size(2), imA:size(3))

  -- Calculate the output RGB chans
  imA[{{1,3},{},{}}]:cmul(imA[{{4},{},{}}]:expandAs(imA[{{1,3},{},{}}]))  -- C_a * alpha_a
  imB[{{1,3},{},{}}]:cmul(imB[{{4},{},{}}]:expandAs(imB[{{1,3},{},{}}]))  -- C_b * alpha_b
  tmp:copy(imA[{{4},{},{}}]):mul(-1):add(1)  -- (1 - alpha_a)
  imB[{{1,3},{},{}}]:cmul(tmp:expandAs(imB[{{1,3},{},{},}])) -- C_b * alpha_b * (1 - alpha_a)
  
  -- C_a * alpha_a + C_b * alpha_b * (1 - alpha_a)
  ret[{{1,3},{},{}}]:copy(imA[{{1,3},{},{}}]):add(imB[{{1,3},{},{}}])

  -- Calculate the ouptut alpha chan
  ret[4]:copy(imB[4]):cmul(tmp)  -- alpha_b * (1 - alpha_a)
  ret[4]:add(imA[4])  -- alpha_a + alpha_b * (1 - alpha_a)

  return ret
end

function determinant2x2(M)
  assert(M:dim() == 2, "M is not 2D")
  assert(M:size(1) == 2 and M:size(2) == 2, "M is not 2x2")
  return M[{1,1}] * M[{2,2}] - M[{1,2}] * M[{2,1}]
end

function determinant3x3(M)
  assert(M:dim() == 2, "M is not 2D")
  assert(M:size(1) == 3 and M:size(2) == 3, "M is not 3x3")
  return M[{1,1}] * (M[{3,3}] * M[{2,2}] - M[{3,2}] * M[{2,3}]) -
         M[{2,1}] * (M[{3,3}] * M[{1,2}] - M[{3,2}] * M[{1,3}]) +
         M[{3,1}] * (M[{2,3}] * M[{1,2}] - M[{2,2}] * M[{1,3}])
end

function inverse2x2(M_inv, M)
  assert(M:dim() == 2, "M is not 2D")
  assert(M:size(1) == 2 and M:size(2) == 2, "M is not 2x2")
  local det = determinant2x2(M)
  if (math.abs(det) < 1e-6) then
    error('Matrix is singular.  No inverse exists.')
  end
  M_inv[{1,1}] = M[{2,2}]
  M_inv[{2,2}] = M[{1,1}]
  M_inv[{1,2}] = -M[{1,2}]
  M_inv[{2,1}] = -M[{2,1}]
  M_inv:mul(1/det)
end

function inverse3x3(M_inv, M)
  assert(M:dim() == 2, "M is not 2D")
  assert(M:size(1) == 3 and M:size(2) == 3, "M is not 3x3")
  local det = determinant3x3(M)
  if (math.abs(det) < 1e-6) then
    error('Matrix is singular.  No inverse exists.')
  end
  M_inv[{1,1}] =   M[{3,3}] * M[{2,2}] - M[{3,2}] * M[{2,3}]
  M_inv[{1,2}] = -(M[{3,3}] * M[{1,2}] - M[{3,2}] * M[{1,3}])
  M_inv[{1,3}] =   M[{2,3}] * M[{1,2}] - M[{2,2}] * M[{1,3}]
  M_inv[{2,1}] = -(M[{3,3}] * M[{2,1}] - M[{3,1}] * M[{2,3}])
  M_inv[{2,2}] =   M[{3,3}] * M[{1,1}] - M[{3,1}] * M[{1,3}]
  M_inv[{2,3}] = -(M[{2,3}] * M[{1,1}] - M[{2,1}] * M[{1,3}])
  M_inv[{3,1}] =   M[{3,2}] * M[{2,1}] - M[{3,1}] * M[{2,2}]
  M_inv[{3,2}] = -(M[{3,2}] * M[{1,1}] - M[{3,1}] * M[{1,2}])
  M_inv[{3,3}] =   M[{2,2}] * M[{1,1}] - M[{2,1}] * M[{1,2}] 
  M_inv:mul(1/det)
end


-- Helper function to create a rotation matrix
function rmat(mat, deg, s)
  local r = deg/180*math.pi
  mat[{1,1}] = s * math.cos(r)
  mat[{1,2}] = -s * math.sin(r)
  mat[{2,1}] = s * math.sin(r)
  mat[{2,2}] = s * math.cos(r)
end

function rmat3(mat, deg, s)
  local r = deg/180*math.pi
  mat[{1,1}] = s * math.cos(r)
  mat[{1,2}] = -s * math.sin(r)
  mat[{1,3}] = 0
  mat[{2,1}] = s * math.sin(r)
  mat[{2,2}] = s * math.cos(r)
  mat[{2,3}] = 0
  mat[{3,1}] = 0
  mat[{3,2}] = 0
  mat[{3,3}] = 1
end

function tmat3(mat, tx, ty)
  mat[{1,1}] = 1
  mat[{1,2}] = 0
  mat[{1,3}] = ty
  mat[{2,1}] = 0
  mat[{2,2}] = 1
  mat[{2,3}] = tx
  mat[{3,1}] = 0
  mat[{3,2}] = 0
  mat[{3,3}] = 1
end

function distortPos2D(labels, deg_rot, scale, trans_v_pix, 
  trans_u_pix, width, height)

  local Mrot = torch.FloatTensor(3,3)
  local Mtrans = torch.FloatTensor(3,3)

  local num_coeff = labels:size(2)
  rmat3(Mrot, deg_rot, 1/scale)
  tmat3(Mtrans, trans_u_pix, trans_v_pix)
  local vec = torch.FloatTensor(3)
 
  M_inv = torch.inverse(torch.mm(Mtrans, Mrot))
  
  -- a (0,0) joint position indicates the joint doesn't exist (or is occluded)
  local zero_mask = labels[{1,{}}]:gt(0):float()
  --labels[{1,{}}] = labels[{1,{}}] / width  -- 0 to 1
  --labels[{2,{}}] = labels[{2,{}}] / height  -- 0 to 1
  --labels:add(-0.5)
  --labels:mul(2)
  labels[{1,{}}]:add(-w/2)
  labels[{2,{}}]:add(-h/2)
  for i = 1, num_coeff do
    vec[1] = labels[{2,i}]  -- v
    vec[2] = labels[{1,i}]  -- u
    vec[3] = 1
    vec = torch.mv(M_inv, vec)
    labels[{1,i}] = vec[2]  -- u
    labels[{2,i}] = vec[1]  -- v
  end
  --labels:add(1)  -- 0 to 2
  --labels:mul(0.5)  -- 0 to 1
  --labels[{1,{}}] = labels[{1,{}}] * width  -- 0 to w
  --labels[{2,{}}] = labels[{2,{}}] * height  -- 0 to h
  labels[{1,{}}]:add(w/2)
  labels[{2,{}}]:add(h/2)  

  -- Make sure occluded joints stay occluded
  labels[{1,{}}]:cmul(zero_mask)
  labels[{2,{}}]:cmul(zero_mask)

end

local colors = {{0,1,0}, {0,0,1}, {1,1,0}, {1,0,1}, {0,1,1}, {0.5,0,0}, 
  {0,0.5,0}, {0,0,0.5}, {0.5,0,0.5}, {0.75,0.75,0}, {0,0.5,0.5}, {0.5,1,0}, 
  {0.5,0,1}, {0.5,1,1,}, {1,0.5,0}, {0,0.5,1}, {1,0.5,1}, {1,0,0.5}, {0,1,0.5}, 
  {1,1,0.5}, {0.5,0.5,0}}

function round(num)
  local under = math.floor(num)
  local upper = math.floor(num) + 1
  local underV = -(under - num)
  local upperV = upper - num
  if (upperV > underV) then
    return under
  else
    return upper
  end
end

function drawLabels(img, uv_labels, w, h, start_color, sigma)
  start_color = start_color or 1
  local max_val = img:max()
  local min_val = img:min()
  local amp = math.max(max_val - min_val, 1e-3)
  local k
  if (uv_labels:dim() == 1) then
    nlabels = 1
  else
    nlabels = uv_labels:size(2)
  end
  for k = 1, nlabels do
    -- TODO: This will fail if all the images are not the same size
    local uv_pos
    if (uv_labels:dim() == 1) then
      uv_pos = { uv_labels[1], uv_labels[2] }
    else
      uv_pos = { uv_labels[{1,k}], uv_labels[{2,k}] }
    end

    local color = {}
    local icolor = math.mod(k-1+start_color, #colors) + 1
    
    for c = 1, 3 do
      table.insert(color,
        min_val + colors[icolor][c] * amp)
    end

    -- Draw a decaying gaussian centered at the pixel
    -- For consistancy, I'm using the code from image.gaussian
    sigma = sigma or 0.005
    
    local mean_v = (uv_pos[2]+0.5)
    local mean_u = (uv_pos[1]+0.5)

    over_sigmau = 1.0 / (sigma * w)  -- Precalculate
    over_sigmav = 1.0 / (sigma * w) 

    local u_min = math.min(math.max(math.floor(mean_u - 3 * sigma * w),1), w)
    local u_max = math.min(math.max(math.floor(mean_u + 3 * sigma * w),1), w)
    local v_min = math.min(math.max(math.floor(mean_v - 3 * sigma * h),1), h)
    local v_max = math.min(math.max(math.floor(mean_v + 3 * sigma * h),1), h)

    for v = v_min, v_max do
     for u = u_min, u_max do
        du = (u - mean_u) * over_sigmau
        dv = (v - mean_v) * over_sigmav
        amp = math.exp(-((du*du*0.5) + (dv*dv*0.5)))
        img[{1,v,u}] = amp * color[1] + (1 - amp) * img[{1,v,u}]
        img[{2,v,u}] = amp * color[2] + (1 - amp) * img[{2,v,u}]
        img[{3,v,u}] = amp * color[3] + (1 - amp) * img[{3,v,u}]
      end
    end
  end
end

