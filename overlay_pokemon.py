import cv2
import numpy as np
import extract_pokemon

# overlay a pokemon on top of the original image
def overlay_pokemon(img, x, y, value):

    match value:
        case 0:
            return img
        case 1:
            pokemon = cv2.imread('data/tepig.png', cv2.IMREAD_UNCHANGED)    #IMREAD_UNCHANGED preserves alpha
        case 2:
            pokemon = cv2.imread('data/totodile.png', cv2.IMREAD_UNCHANGED)
        case 3:
            pokemon = cv2.imread('data/bulbasaur.png', cv2.IMREAD_UNCHANGED)
        case _:
            return img
    
    pokemon = extract_pokemon.resize_img(pokemon, 20)   # reusing function to resize img to 20% of original
    mask = pokemon[:, :, 3] / 255.0 # create alpha mask
    pokemon_rgb = cv2.cvtColor(pokemon,cv2.COLOR_RGBA2RGB)  # remove alpha channel from pokemon

    mask = np.repeat(mask[:,:,np.newaxis], 3, axis=2)   # expand mask dims to (n,n,3)
    result = overlay_image_alpha(img, pokemon_rgb, x, y, mask)

    return result

# idk how this works, source: https://stackoverflow.com/questions/14063070/overlay-a-smaller-image-on-a-larger-image-python-opencv
def overlay_image_alpha(img, img_overlay, x, y, alpha_mask=None):
    """Overlay `img_overlay` onto `img` at (x, y) and blend using optional `alpha_mask`.

    `alpha_mask` must have same HxW as `img_overlay` and values in range [0, 1].
    """

    if y < 0 or y + img_overlay.shape[0] > img.shape[0] or x < 0 or x + img_overlay.shape[1] > img.shape[1]:
        y_origin = 0 if y > 0 else -y
        y_end = img_overlay.shape[0] if y < 0 else min(img.shape[0] - y, img_overlay.shape[0])

        x_origin = 0 if x > 0 else -x
        x_end = img_overlay.shape[1] if x < 0 else min(img.shape[1] - x, img_overlay.shape[1])

        img_overlay_crop = img_overlay[y_origin:y_end, x_origin:x_end]
        alpha = alpha_mask[y_origin:y_end, x_origin:x_end] if alpha_mask is not None else None
    else:
        img_overlay_crop = img_overlay
        alpha = alpha_mask


    y1 = max(y, 0)
    y2 = min(img.shape[0], y1 + img_overlay_crop.shape[0])

    x1 = max(x, 0)
    x2 = min(img.shape[1], x1 + img_overlay_crop.shape[1])

    img_crop = img[y1:y2, x1:x2]
    if alpha.shape != img_crop.shape != img_overlay_crop.shape:
        return img
    img_crop[:] = alpha * img_overlay_crop + (1.0 - alpha) * img_crop if alpha is not None else img_overlay_crop

    return img