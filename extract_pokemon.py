# failed attempts at extracting pokemon image from the card


import cv2
import numpy as np

def extract_pokemon_from_image(img_path):
    # read the image
    big_img = cv2.imread(img_path)

    # resize to 30% of original
    img = resize_img(big_img, 30)

    # define the foreground rectangle
    tlx = 100
    tly = 60
    lenx = 300
    leny = 300
    rect = (tlx, tly, lenx, leny)

    # apply grabcut
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # create a binary mask
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

    # apply the mask to the original image to extract the foreground
    foreground = img * mask2[:, :, np.newaxis]

    # draw rectangle around box that was used for extraction
    # (for testing purposes)
    # calculate top left and bottom right corners of box
    tl = (tlx, tly)
    brx = tlx + lenx
    bry = tly + leny
    br = (brx, bry)

    #draw
    cv2.rectangle(img, tl, br, color=(0, 225, 0), thickness=2)
    cv2.rectangle(foreground, tl, br, color=(0, 225, 0), thickness=2)

    return img, foreground


def resize_img(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    return resized

def test_extract():
    # extract from all images
    bulbasaur, ebulbasaur = extract_pokemon_from_image('data/bulbasaur_card.jpg')
    tepig, etepig = extract_pokemon_from_image('data/tepig_card.jpg')
    totodile, etotodile = extract_pokemon_from_image('data/totodile_card.jpg')

    # display all new images
    cv2.imshow("bulbasaur", bulbasaur)
    cv2.imshow("extracted_bulbasaur", ebulbasaur)
    cv2.imshow("tepig", tepig)
    cv2.imshow("extracted_tepig", etepig)
    cv2.imshow("totodile", totodile)
    cv2.imshow("extracted_totodile", etotodile)

    # arrange windows
    cv2.moveWindow("bulbasaur", 0, 0)
    cv2.moveWindow("extracted_bulbasaur", 0, 500)
    cv2.moveWindow("tepig", bulbasaur.shape[1], 0)
    cv2.moveWindow("extracted_tepig", bulbasaur.shape[1], 500)
    cv2.moveWindow("totodile", bulbasaur.shape[1] + tepig.shape[1], 0)
    cv2.moveWindow("extracted_totodile", bulbasaur.shape[1] + tepig.shape[1], 500)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def segment_image(img_path):
     # read the image
    big_img = cv2.imread(img_path)

    # resize to 30% of original
    full_img = resize_img(big_img, 30)

    # extract important part of card
    img = crop_img(full_img)

    # convert to grayscale
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # generate threshold
    val = np.mean(img_gray)
    _,thresh = cv2.threshold(img_gray, val, 255, cv2.THRESH_BINARY_INV)

    # detect edges using canny edge detector
    edges = cv2.dilate(cv2.Canny(thresh,0,255),None)

    # detect contours to create mask
    cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
    mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    masked = cv2.drawContours(mask, [cnt],-1, 255, -1)

    # segment the regions
    dst = cv2.bitwise_and(img, img, mask=mask)
    # segmented = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

    return full_img, dst


# crop the rectangle containing pokemon from the rest of the card
def crop_img(img):
    return img[60:360, 100:400]

def test_segment():
    bulbasaur, ebulbasaur = segment_image('data/bulbasaur_card.jpg')
    tepig, etepig = segment_image('data/tepig_card.jpg')
    totodile, etotodile = segment_image('data/totodile_card.jpg')

    # display all new images
    cv2.imshow("bulbasaur", bulbasaur)
    cv2.imshow("segmented_bulbasaur", ebulbasaur)
    cv2.imshow("tepig", tepig)
    cv2.imshow("segmented_tepig", etepig)
    cv2.imshow("totodile", totodile)
    cv2.imshow("segmented_totodile", etotodile)

    # arrange windows
    cv2.moveWindow("bulbasaur", 0, 0)
    cv2.moveWindow("segmented_bulbasaur", 0, 500)
    cv2.moveWindow("tepig", bulbasaur.shape[1], 0)
    cv2.moveWindow("segmented_tepig", bulbasaur.shape[1], 500)
    cv2.moveWindow("totodile", bulbasaur.shape[1] + tepig.shape[1], 0)
    cv2.moveWindow("segmented_totodile", bulbasaur.shape[1] + tepig.shape[1], 500)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
