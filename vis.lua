dofile("init.lua")

normalize_weight = function(nn_module, convFlag)
   local w = nn_module.weight:clone():type('torch.FloatTensor')
   if not convFlag then
      -- F.C
      local nImg = w:size(1)
      local nSize = w:size(2)
      local nSizeSqrt = math.floor(math.sqrt(nSize))
      w = w[{ {}, {1,nSizeSqrt^2} }]
      w = w:reshape(nImg, nSizeSqrt, nSizeSqrt)
      for i = 1, nImg do
         w[i]:div(w[i]:norm())
      end
      w_norm = w:clone()
   else
      -- Convolution
      for i = 1, w:size(1) do
         for j = 1, w:size(2) do
            w[{ i, j, {}, {} }]:div(w[{ i, j, {}, {} }]:norm())
         end
      end
      w_norm = torch.squeeze(w)
   end
   return w_norm
end


function string.ends(String,End)
   return End=='' or string.sub(String,-string.len(End))==End
end


form_weight_table = function(dirname)
   wTb = {}
   p = io.popen('find ' .. dirname .. ' -maxdepth 1 -type f')
   for fl in p:lines() do
      if string.ends(fl, '.net') then
         -- It is a model file
         ml = torch.load(fl)
         wTb[fl] = normalize_weight(ml:get(2), true)
      end
   end
   return wTb
end


visualize = function(weight_table)
   if gfx ~= nil then
      for k, v in pairs(weight_table) do
         gfx.image(v, {zoom=5, legend=k})
      end
   else
      torch.save('Results/filter.t7', weight_table)
   end
end

visualize(form_weight_table('Results'))
