import matplotlib.pyplot as plt


def simple_im_show(img, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

def simple_im_show2(img, mask, figsize=(10, 10)):
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    img_cmap = 'gray' if len(img.shape) == 2 else None
    mask_cmap = 'gray' if len(img.shape) == 2 else None
    ax[0].imshow(img, cmap=img_cmap)
    ax[1].imshow(mask, cmap=mask_cmap)
    ax[0].axis('off')
    ax[1].axis('off')
    plt.show()