import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

lesions = ["PAT_29_40_561_mask.npy"]
fig,ax = plt.subplots(3,2,figsize=(9,16))
for lesion in lesions:
    
    mask = np.load("../../segmentation/masks/"+lesion)
    mask[mask>0] = 1

    hflip = np.flipud(mask)
    diff_hflip = mask-hflip
    vflip = np.fliplr(mask)
    diff_vflip = mask-vflip

    rot = rotate(mask,angle=45)
    rothflip = np.flipud(rot)
    diff_rothflip = rot-rothflip
    rotvflip = np.fliplr(rot)
    diff_rotvflip = rot-rotvflip

    # diff_hflip[((mask==1) == (diff_hflip == 0))==True] = 0.5
    # diff_vflip[((mask==1) == (diff_vflip == 0))==True] = 0.5
    # diff_rothflip[((mask==1) == (diff_rothflip == 0))==True] = 0.5
    # diff_rotvflip[((mask==1) == (diff_rotvflip == 0))==True ] = 0.5
        
    
    ax[0][0].imshow(mask,cmap="gray")
    ax[1][0].imshow(diff_hflip,cmap="gray")
    ax[1][1].imshow(diff_vflip,cmap="gray")
    ax[0][1].imshow(rot,cmap="gray")
    ax[2][0].imshow(diff_rothflip,cmap="gray")
    ax[2][1].imshow(diff_rotvflip,cmap="gray")
    ax[0][0].set_title("Original")
    ax[1][0].set_title("Horizontal Flip")
    ax[1][1].set_title("Vertical Flip")
    ax[0][1].set_title("Rotation")
    ax[2][0].set_title("Rotation + Horizontal Flip")
    ax[2][1].set_title("Rotation + Vertical Flip")
    
    fig.tight_layout(pad=10)
    plt.show()