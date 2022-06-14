import matplotlib as plt
import numpy as np
import cv2

def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image


def replace_mask(img, img_bg, mask):
    """
    Use mask to replace pixels in img with img_bg
    - img:      RGB images
    - img_bg:   background image 
    """

    res = cv2.bitwise_and(img, img, mask = mask)
    f = img - res   # set masked pixels to 0
    f = np.where(f == 0, img_bg, f) # fill the masked pixels with img_bg

    return f

def largest_connected_comopnent(mask):

    #-----------------------------------------------------
    # find the largest connected component of binary image
    #-----------------------------------------------------


    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # connectedComponentswithStats yields every seperated component with information on each of them, such as size
    # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    idx_max_comp = np.argmax(sizes) # index for largest connected component
    idx_mask = np.where(output == (idx_max_comp+1)) # location of the connected components in the image
    mask2 = np.zeros_like(mask)
    mask2[idx_mask] = 1 

    return mask2

def fill_binary(img_gray):

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    res = cv2.morphologyEx(img_gray,cv2.MORPH_OPEN,kernel)

    return res

def fill_largest_component(mask):
    """Find largest component and fill inside
    """
    
    mask_largest = largest_connected_comopnent(mask) # detect blanket as white
    mask_largest_inv = 1-mask_largest

    # detect outside of blanket as white
    # ASSUME This assumes that outside blanket largest non-masked area.
    # TODO: fix this assumption
    mask_clean_inv = largest_connected_comopnent(mask_largest_inv)

    return 1 - mask_clean_inv


filename = 'IMG_8837'
path = '/home/sarahl/Desktop/seg_color/seg_images/' + filename + '.jpg'
img = cv2.imread(path)


#-----------------------------------------------------
# Use color range to mask image
#-----------------------------------------------------

u_color = np.array([200,250,230]) #([255, 255, 255])
l_color = np.array([60,30,96])
mask = cv2.inRange(img, l_color, u_color)    # one channel with 1's and 0's


# create solid green image
img_solid = create_blank(img.shape[1],img.shape[0],(0,0,0))

# use mask to 
img_masked1 = replace_mask(img, img_solid, mask)
fname_out1 = '/home/sarahl/Desktop/seg_color/seg_images_results/' + filename + '_masked1.jpg'
# fname_out1 = '/home/sarahl/Desktop/seg_color/seg_images_results/seg_images_mask1/' + filename + '_masked1.jpg'
cv2.imwrite(fname_out1, img_masked1)


#-----------------------------------------------------
# find the largest connected component of binary image
#-----------------------------------------------------


mask2 = fill_largest_component(mask)

img_masked2 = replace_mask(img, img_solid, mask2)

#fname_out3 = '/home/sarahl/Desktop/seg_color/seg_images_results/seg_images_mask2/' + filename + '_masked2.jpg'
fname_out3 = '/home/sarahl/Desktop/seg_color/' + filename + '_masked2.jpg'
cv2.imwrite(fname_out3,img_masked2)
