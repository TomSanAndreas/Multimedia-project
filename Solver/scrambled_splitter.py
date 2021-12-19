#bron: https://stackoverflow.com/questions/55169645/square-detection-in-image
import cv2
import numpy as np

"""Source:  https://stackoverflow.com/questions/45613544/python-opencv-cannot-change-pixel-value-of-a-picture
            https://www.geeksforgeeks.org/python-grayscaling-of-images-using-opencv/
            rotatie/crop: https://newbedev.com/how-to-straighten-a-rotated-rectangle-area-of-an-image-using-opencv-in-python
            crop: https://stackoverflow.com/questions/13538748/crop-black-edges-with-opencv
            roteren hardcoded: https://learnopencv.com/image-rotation-and-translation-using-opencv/#image-rotation
    
    functie werkt 100% bij 8/9 van de figuren. (Bij de figuren 02 herkent hij de randen maar ook de ogen van de kat)
"""

def crop_rectangle(img, rect):
    # get the parameter of the small rectangle
    center = rect[0]
    size = rect[1]
    angle = rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))
    # get row and col num in img
    height, width = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D(center, angle, 1)
    img_rot = cv2.warpAffine(img, M, (width, height))
    img_crop = cv2.getRectSubPix(img_rot, size, center)
    return img_crop, img_rot

def crop_rect_help(image):
    gray_rot = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_rot, 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # randen tekenen op figuur
    img_crop, img_rot = crop_rectangle(image, rect)
    new_size = (int(img_rot.shape[1] / 2), int(img_rot.shape[0] / 2))
    img_rot_resized = cv2.resize(img_rot, new_size)
    new_size = (int(image.shape[1] / 2)), int(image.shape[0] / 2)
    img_resized = cv2.resize(image, new_size)
    return img_crop

def knip_tegels(image: np.ndarray, dimensions: tuple[int, int]) -> list[np.ndarray]:
    # ---Figuur inlezen + aantal stukken en parameters bepalen (VIA PARAMETERS)---
    aantal_h, aantal_v = dimensions

    #---stukken splitsen---
    pieces = [[None for _ in range(aantal_h)] for _ in range(aantal_v)]
    pieces_res = [[None for _ in range(aantal_h)] for _ in range(aantal_v)]

    b = int(image.shape[0]/aantal_v)
    l = int(image.shape[1]/aantal_h)
    pieceno = 0
    for i in range(aantal_v):
        for j in range(aantal_h):
            pieceno = pieceno+1
            pieces[i][j] = image[(i*b):((i+1)*b), (j*l):((j+1)*l)]
            #afmetingen controleren
            pieces_res[i][j] = crop_rect_help(pieces[i][j])
    # stukken flattenen naar 1 lange lijst
    pieces = [img for imgs in pieces_res for img in imgs]
    return pieces

# DEMO
# if __name__ == '__main__':
#     from matplotlib import pyplot as plt
#     filename = 'Documentatie/Puzzels/Tiles_scrambled/tiles_scrambled_2x2_00.png'
#     image = cv2.imread(filename)
#     index = filename.find('x')
#     aantal_h = int(filename[index-1])
#     aantal_v = int(filename[index+1])
#     dimensions = [aantal_v,aantal_h]
#     pieces = knip_tegels(image, dimensions)

#     fig = plt.figure()
#     index = 0
#     for piece in pieces:
#         index += 1
#         ax = fig.add_subplot(aantal_h, aantal_v, index)
#         ax.imshow(piece[:,:,::-1])
#         ax.axis("off")
#     plt.show()
