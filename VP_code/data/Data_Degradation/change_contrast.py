from PIL import Image, ImageEnhance 
import albumentations as A
import numpy as np
def color_jitter(image, verbose=False):

    transform=A.ColorJitter(brightness=[0.7,0.8], contrast=[0.9,1.0], saturation=[0.9,1.0], hue=0.0, always_apply=True, p=1.0)
    image=np.array(image)
    jittered=transform(image=image)['image']

    x=Image.fromarray(jittered.astype('uint8')).convert('L')
    return x

im = Image.open("00011.png")
# enhancer = ImageEnhance.Contrast(im)
# enhanced_im = enhancer.enhance(1.0)
enhanced_im=color_jitter(im)
enhanced_im.save("enhanced_see.png")