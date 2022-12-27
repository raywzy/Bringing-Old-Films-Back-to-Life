from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import random
import math
import os

def texture_augmentation(texture_image, target_h, target_w):

    # h,w = texture_image.shape
    ## Random resize
    texture_image = texture_image.resize((random.randint(target_w,target_w*2),random.randint(target_h,target_h*2)))
    w,h = texture_image.size

    ## Random crop
    top = random.randint(0, h - target_h)
    left = random.randint(0, w - target_w)
    croped_texture = texture_image.crop((left,top,left+target_w,top+target_h))
    
    return croped_texture


def texture_augmentation_new(texture_image, target_h, target_w):

    # h,w = texture_image.shape
    ## Random resize
    texture_image = texture_image.resize((random.randint(target_w,target_w*2),random.randint(target_h,target_h*2)))
    w,h = texture_image.size

    ## Random crop
    top = random.randint(0, h - target_h)
    left = random.randint(0, w - target_w)
    croped_texture = texture_image.crop((left,top,left+target_w,top+target_h))
    
    return croped_texture

##TODO: 1. Add rotation [√] 2. Treat 008 [√] 3. Some probability directly resize or random crop [√]
def texture_generator(texture_input, folder_name, target_h=256, target_w=256):

    random_P = random.uniform(0.0, 1.0)
    dilation_group = ['002','003','004','005','006','007','009','012']
    dilation_flag = False
    if folder_name in dilation_group:
      dilation_flag = True

    if random_P < 0.15 or folder_name == '008':
        
        final_texture = texture_input.resize((target_w,target_h),resample=Image.LANCZOS)
        return final_texture

    W,H= texture_input.size

    ## Calculate scratch position
    texture_mean = np.mean(texture_input)
    texture_mask = (texture_input<(texture_mean-15))*1.

    ## Remove the boundary sample
    texture_mask[:,:40]=0.0
    texture_mask[:,-40:]=0.0

    # texture_mask_pil = Image.fromarray(255*texture_mask.astype('uint8')).convert("L")
    # texture_mask_pil.save("mask.png")
    if np.sum(texture_mask)<1:
      final_texture = texture_input.resize((target_w,target_h),resample=Image.LANCZOS)
      return final_texture

    ## Sample anchor point
    anchor_point_y,anchor_point_x = random.sample(list(zip(*np.where(texture_mask))), 1)[0]
    # print("Point should be contained: ")
    # print((anchor_point_y,anchor_point_x))
    # print(texture_mask[anchor_point_y, anchor_point_x])

    ## Sample box size
    bounding_box_size = random.randint(150, 360) ## Original: [130,360]
    # print("Bounding box size: ")
    # print(bounding_box_size)

    ## Sample shift size (between anchor and left_up point)
    shift_x = random.randint(max(bounding_box_size-W+anchor_point_x,0), min(anchor_point_x,bounding_box_size))
    shift_y = random.randint(max(bounding_box_size-H+anchor_point_y,0), min(anchor_point_y,bounding_box_size))
    # print("Shift size: ")
    # print((shift_x,shift_y))

    ## Get left_up point
    left_up_x = anchor_point_x - shift_x
    left_up_y = anchor_point_y - shift_y
    # print("Left_Up: ")
    # print((left_up_x,left_up_y))

    croped_texture = texture_input.crop((left_up_x,left_up_y,left_up_x+bounding_box_size,left_up_y+bounding_box_size))

    ## Add rotation
    rotation_angle = random.randint(0,360)
    rotated_texture = croped_texture.rotate(angle=rotation_angle,expand=True,resample=Image.BILINEAR)
    max_w,max_h = rotatedRectWithMaxArea(bounding_box_size,bounding_box_size,math.radians(rotation_angle))
    final_texture = center_crop(rotated_texture,max_w,max_h)

    final_texture = final_texture.resize((target_w,target_h),resample=Image.LANCZOS)
    
    ## Dilation for specific group
    if dilation_flag:
      dilation_kernel_size = random.randint(0,1)*2+1
      final_texture = final_texture.filter(ImageFilter.MinFilter(dilation_kernel_size))
      # print(dilation_kernel_size)

    ## Change Contrast
    if random.uniform(0.0,1.0)<0.7:
      enhancer = ImageEnhance.Contrast(final_texture)
      final_texture = enhancer.enhance(random.uniform(2.0, 4.0))

    return final_texture

  
## Used for the generate the texture for moving line
def moving_line_texture_generator(texture_input, last_texture, target_h=256, target_w=256):

    if last_texture is None:
      W,H= texture_input.size

      ## Calculate scratch position
      texture_mean = np.mean(texture_input)
      texture_mask = (texture_input<(texture_mean-25))*1.

      ## Remove the boundary sample
      texture_mask[:,:40]=0.0
      texture_mask[:,-40:]=0.0


      ## Sample anchor point
      anchor_point_y,anchor_point_x = random.sample(list(zip(*np.where(texture_mask))), 1)[0]
      # print("Point should be contained: ")
      # print((anchor_point_y,anchor_point_x))
      # print(texture_mask[anchor_point_y, anchor_point_x])

      ## Sample box size
      bounding_box_size = random.randint(150, 360) ## Original: [130,360]
      # print("Bounding box size: ")
      # print(bounding_box_size)

      ## Sample shift size (between anchor and left_up point)
      shift_x = random.randint(max(bounding_box_size-W+anchor_point_x,0), min(anchor_point_x,bounding_box_size))
      shift_y = random.randint(max(bounding_box_size-H+anchor_point_y,0), min(anchor_point_y,bounding_box_size))
      # print("Shift size: ")
      # print((shift_x,shift_y))

      ## Get left_up point
      left_up_x = anchor_point_x - shift_x
      left_up_y = anchor_point_y - shift_y
      # print("Left_Up: ")
      # print((left_up_x,left_up_y))

      croped_texture = texture_input.crop((left_up_x,left_up_y,left_up_x+bounding_box_size,left_up_y+bounding_box_size))

      final_texture = croped_texture.resize((target_w,target_h),resample=Image.LANCZOS)
    
    else:

      random_direction = random.uniform(0,1) > 0.5
      random_distance = random.randint(5,15)

      texture_np = np.array(last_texture)
      # print(random_direction)
      # print(random_distance)
      if random_direction:

        texture_np=np.hstack((texture_np[:,random_distance:],texture_np[:,0-random_distance:]))
      else:
        texture_np=np.hstack((texture_np[:,:random_distance],texture_np[:,:target_w-random_distance]))

      final_texture = Image.fromarray(np.uint8(texture_np) , 'L')

      ## Dilation for specific group
      # if dilation_flag:
      #   dilation_kernel_size = random.randint(0,1)*2+1
      #   final_texture = final_texture.filter(ImageFilter.MinFilter(dilation_kernel_size))
      #   # print(dilation_kernel_size)

      ## Change Contrast
      # if random.uniform(0.0,1.0)<0.7:
      #   enhancer = ImageEnhance.Contrast(final_texture)
      #   final_texture = enhancer.enhance(random.uniform(2.0, 4.0))

    return final_texture

def rotatedRectWithMaxArea(w, h, angle):
  """
  Given a rectangle of size wxh that has been rotated by 'angle' (in
  radians), computes the width and height of the largest possible
  axis-aligned rectangle (maximal area) within the rotated rectangle.
  """
  if w <= 0 or h <= 0:
    return 0,0

  width_is_longer = w >= h
  side_long, side_short = (w,h) if width_is_longer else (h,w)

  # since the solutions for angle, -angle and 180-angle are all the same,
  # if suffices to look at the first quadrant and the absolute values of sin,cos:
  sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
  if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
    # half constrained case: two crop corners touch the longer side,
    #   the other two corners are on the mid-line parallel to the longer line
    x = 0.5*side_short
    wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
  else:
    # fully constrained case: crop touches all 4 sides
    cos_2a = cos_a*cos_a - sin_a*sin_a
    wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

  return wr,hr

def center_crop(pil_image, new_width, new_height):

    width, height = pil_image.size 

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    im = pil_image.crop((left, top, right, bottom))

    return im

def getfilelist(file_path):
    all_file = []
    for dir,folder,file in os.walk(file_path):
        for i in file:
            t = "%s/%s"%(dir,i)
            all_file.append(t)
    all_file = sorted(all_file)
    return all_file

if __name__=='__main__':

    # random.seed(12)
    texture_url = "./data/Data_Degradation/noise_data"

    texture_templates = getfilelist(texture_url)

    selected_texture_url=random.sample(texture_templates,1)[0]
    texture_image = Image.open(selected_texture_url).convert("L")
    folder_name = selected_texture_url.split('/')[-2]
    # texture_np = np.array(texture_image)
    # texture_median = np.mean(texture_np)
    # texture_mask = (texture_np<(texture_median-15))*255.

    # texture_mask_pil = Image.fromarray(texture_mask.astype('uint8')).convert("L")
    # texture_mask_pil.save("test_mean.png")
    print(folder_name)
    final_texture = texture_generator(texture_image,folder_name)

    final_texture.save("see_enhance.png")

    # rotated_texture=texture_image.rotate(angle=100,expand=True,resample=Image.BILINEAR)
    # w,h = rotatedRectWithMaxArea(256,256,math.radians(100))
    # final_texture=center_crop(rotated_texture,w,h)

    # final_texture.save("rotate.png")
    # rotated_texture.save("non_rotate.png")