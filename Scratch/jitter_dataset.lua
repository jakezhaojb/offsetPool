torch.setdefaulttensortype('torch.FloatTensor')
require("image")
dofile('pbar.lua') 
dofile('distort.lua') 

-- Dataset settings for training
data_dir = '/misc/vlgscratch3/LecunGroup/provodin/lagr/pp' 
image_dir = data_dir..'/quality9075/LAGR2014_img_train/'
save_dir = '/home/goroshin/Projects/TAE/Data/ImageNet/'
list = torch.load(data_dir.."/index/train.t7b")
jitter_dataset = torch.Tensor(nsample,njitters+1,3,height,width) 

--options 
nsample = 10000
njitters = 10 
height = 114 
width = 114 
rot_range = {-10,10} 
dx_range = {-0.01,0.01}
dy_range = {-0.01,0.01}  
scale_range = {-0.2, 0.2} 
xsheer_range = {-0.2,0.2} 
ysheer_range = {-0.2,0.2} 
crop_method = 'center' 

dofile("jitter_utils.lua")

--main 
for k = 1,nsample do 
   
    progress(k,nsample) 
    
    class_idx = math.random(#list.files) 
    file_idx = math.random(#list.files[class_idx])  

    I = load_image(image_dir..list.files[class_idx][file_idx]) 
    Ic = crop(I,height,width,crop_method) 
    jitter_dataset[k][1]:copy(Ic) 
    
    for i = 2,njitters+1 do 
        
        p1 = math.random() 
        p2 = math.random() 
        p3 = math.random() 
        p4 = math.random() 
        p5 = math.random() 
        p6 = math.random() 
        
        rot = p1*rot_range[1] + (1-p1)*rot_range[2] 
        scale = p2*scale_range[1] + (1-p2)*scale_range[2] 
        dx = p3*dx_range[1] + (1-p3)*dx_range[2] 
        dy = p4*dy_range[1] + (1-p4)*dy_range[2] 
        xsheer = p5*xsheer_range[1] + (1-p5)*xsheer_range[2] 
        ysheer = p6*ysheer_range[1] + (1-p6)*ysheer_range[2] 

        Id = distort(I, rot, scale, dx, dy, xsheer, ysheer)  
        Idc = crop(Id, height, width, 'center')  
        
        jitter_dataset[k][i]:copy(Idc) 

    end
    collectgarbage() 
end 

torch.save(save_dir..'ImageNet_jitter_small.t7',jitter_dataset) 

--I = image.toDisplayTensor({input = jitter_dataset:resize((n+1)*njitters,3,114,114), nrow = njitters+1, padding = 1})
--image.save('./sample.png',I) 


