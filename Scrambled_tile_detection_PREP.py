#bron: https://stackoverflow.com/questions/55169645/square-detection-in-image
import cv2
import numpy as np
# import sys

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
    #print("width: {}, height: {}".format(width, height))
    M = cv2.getRotationMatrix2D(center, angle, 1)
    img_rot = cv2.warpAffine(img, M, (width, height))
    img_crop = cv2.getRectSubPix(img_rot, size, center)
    return img_crop, img_rot

def crop_rect_help(image):
    gray_rot = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_rot, 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(len(contours[0]))
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    rect = cv2.minAreaRect(cnt)
    #print("rect: {}".format(rect))
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # print("bounding box: {}".format(box))
    # randen tekenen op figuur
    # cv2.drawContours(pieces[0][0], [box], 0, (0, 0, 255), 2)
    img_crop, img_rot = crop_rectangle(image, rect)
    #print("size of original img: {}".format(pieces[0][0].shape))
    #print("size of rotated img: {}".format(img_rot.shape))
    #print("size of cropped img: {}".format(img_crop.shape))
    new_size = (int(img_rot.shape[1] / 2), int(img_rot.shape[0] / 2))
    img_rot_resized = cv2.resize(img_rot, new_size)
    new_size = (int(image.shape[1] / 2)), int(image.shape[0] / 2)
    img_resized = cv2.resize(image, new_size)
    # cv2.imshow("original contour", img_resized)
    # cv2.imshow("rotated image", img_rot_resized)
    # cv2.imshow("cropped_box", img_crop)
    return img_crop

def knip_tegels(image, dimensions):
    # ---Figuur inlezen + aantal stukken en parameters bepalen (VIA PARAMETERS)---
    # filename = 'Documentatie/Puzzels/Tiles_scrambled/tiles_scrambled_3x3_00.png'
    # image = cv2.imread(filename)
    # index = filename.find('x')
    # aantal_h = int(filename[index-1])
    # aantal_v = int(filename[index+1])

    aantal_h = dimensions[1]
    aantal_v = dimensions[0]
    print(aantal_h)
    print(aantal_v)

    #figuur converteren naar grijswaarden
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #optioneel: lengte en breedte bepalen (nodig voor methode 2)
    # (row,col) = gray.shape[0:2]
    # print(row, col)
    # splits = row/2

    #---stukken splitsen---
    pieces = [[0]*aantal_h]*aantal_v
    pieces_res = [[0]*aantal_h]*aantal_v
    # print(pieces)
    b = int(image.shape[0]/aantal_h)
    l = int(image.shape[1]/aantal_v)
    pieceno = 0
    for i in range(aantal_v):
        for j in range(aantal_h):
            pieceno = pieceno+1
            #print(pieceno)
            pieces[i][j] = image[(i*b):((i+1)*b), (j*l):((j+1)*l)]
            #afmetingen controleren
            #print(str(i*b) + " " + str((i+1)*b) + " " + str(j*l) + " " + str((j+1)*l))
            #cv2.imshow("Piece no." + str(pieceno), pieces[i][j])
            pieces_res[i][j] = crop_rect_help(pieces[i][j])
            # Resultaat tonen
            cv2.imshow("Piece no." + str(pieceno), pieces_res[i][j])

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# DEMO
if __name__ == '__main__':
    filename = 'Documentatie/Puzzels/Tiles_scrambled/tiles_scrambled_5x5_06.png'
    image = cv2.imread(filename)
    index = filename.find('x')
    aantal_h = int(filename[index-1])
    aantal_v = int(filename[index+1])
    dimensions = [aantal_v,aantal_h]
    pieces = knip_tegels(image, dimensions)






"""
#Oude crop methode
    # # for c in cnt:
    # #     cv2.drawContours(pieces[0][0],[c], 0, (0,255,0), 2)
    # crop = pieces[0][0][y+1:y+h-1,x+2:x+w-2]
    # cv2.imshow("test", crop)

#---OUDE METHODE voor randdetectie---
#puzzeltegels witmaken vanuit grijswaardenbeeld (methode1)
# notblack=np.where((gray[:,:]!=0))
# gray[notblack]=(255)

#(methode2)
# for i in range(row):
#     for j in range(col):
#         # Find the average of the BGR pixel values
#         if (gray[i, j] != 0):
#             gray[i, j] = 255

#tegels witmaken vanuit inputfiguur
# notblack=np.where((image[:,:,0]!=0) & (image[:,:,1]!=0) & (image[:,:,2]!=0))
# image[notblack]=(255,255,255)

#tussenresultaat tonen
#cv2.imshow('test', gray)
# cv2.waitKey()
# cv2.destroyAllWindows()

#Canny toepassen + randen van de tegels tekenen
# canny = cv2.Canny(gray, 130, 255, 1)
# # hough = cv2.HoughLines(canny, 1, np.pi / 180, 150, None, 0, 0)
# cnts = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# for c in cnts:
#     cv2.drawContours(image,[c], 0, (0,255,0), 2)



Hardoded ter test:
#hardcoded voor 2x2
# pieceRT = image[:b,l+1:]
# pieceLT = image[:b,:l]
# pieceLB = image[b+1:,:l]
# pieceRB = image[b+1:,l+1:]
# cv2.imshow("Left Top", crop_rect_help(pieceLT))
# cv2.imshow("Right Top", crop_rect_help(pieceRT))
# cv2.imshow("Left Bottom", crop_rect_help(pieceLB))
# cv2.imshow("Right Bottom", crop_rect_help(pieceRB))

# cv2.imshow("piece1", pieces_res[0][0])
# cv2.imshow("piece2", pieces_res[0][1])
# cv2.imshow("piece3", pieces_res[1][0])
# cv2.imshow("piece4", pieces_res[1][1])

#rotatie voor piece0
# height, width = pieces[0][0].shape[:2]
# center = (width/2, height/2)
# print(center)
# rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=17, scale=1)
# rotated_image = cv2.warpAffine(src=pieces[0][0], M=rotate_matrix, dsize=(width, height))
# cv2.imshow('Original image', pieces[0][0])
# cv2.imshow('Rotated image', rotated_image)

Oude code (backup/magweg)
1.
--- Detecteren aantal stukken van een tiled (en niet scrambled) puzzel ---
edges = cv2.Canny(img, 250, 400)
# threshold dynamisch regelen tot het aantal gevonden lines groot genoeg is
lines = [[], []]
threshold = 2000
while (len(lines[0]) < 1 or len(lines[1]) < 1) and threshold > 50:
    threshold -= 45
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold, None, 0, 0)
    lines = [] if lines is None else lines
    # enkel lijnen met een theta = 0 en 90° zijn relevant,
    # dus verkeerde wegfilteren
    lines = [[line[0] for line in lines if 0 <= line[0][1] <= 0.001], [line[0] for line in lines if np.pi / 2 - 0.0005 <= line[0][1] <= np.pi / 2 + 0.0005]]
# de lijnen in lines kunnen gebruikt worden om te weten hoeveel stukken er
# zijn (n x m)
# horizontaal aantal stukken bepalen
n_h = 10
for line in lines[0]:
    for i in range(-1, 2):
        n_horizontal = 1
        ratio = (line[0] + i) / img.shape[1]
        while n_horizontal < 10 and ratio * n_horizontal != int(ratio * n_horizontal):
            n_horizontal += 1
        if n_h > n_horizontal:
            n_h = n_horizontal
if n_h == 10:
    raise RuntimeError("Aantal horizontale stukken werd niet correct gedetecteerd!")
# verticaal aantal stukken bepalen
n_v = 10
for line in lines[1]:
    for i in range(-1, 2):
        n_vertical = 1
        ratio = (line[0] + i) / img.shape[0]
        while n_vertical < 10 and ratio * n_vertical != int(ratio * n_vertical):
            n_vertical += 1
        if n_v > n_vertical:
            n_v = n_vertical
if n_v == 10:
    raise RuntimeError("Aantal verticale stukken werd niet correct gedetecteerd!")
    
2.# image = cv2.imread('Documentatie/Puzzels/Tiles_scrambled/tiles_scrambled_2x2_01.png')
#
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blur = cv2.medianBlur(gray, 5)
# sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
# sharpen = cv2.filter2D(blur, -1, sharpen_kernel)
#
# thresh = cv2.threshold(sharpen,160,255, cv2.THRESH_BINARY_INV)[1]
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
#
# cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#
# min_area = 100
# max_area = 1500
# image_number = 0
# for c in cnts:
#     area = cv2.contourArea(c)
#     if area > min_area and area < max_area:
#         x,y,w,h = cv2.boundingRect(c)
#         ROI = image[y:y+h, x:x+w]
#         cv2.imwrite('ROI_{}.png'.format(image_number), ROI)
#         cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
#         image_number += 1
#
# cv2.imshow('sharpen', sharpen)
# cv2.imshow('close', close)
# cv2.imshow('thresh', thresh)
# cv2.imshow('image', image)
# cv2.waitKey()

# img = cv2.imread('Documentatie/Puzzels/Tiles_scrambled/tiles_scrambled_2x2_00.png')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray,50,150,apertureSize = 3)
# minLineLength = 100
# maxLineGap = 10
# lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
# for x1,y1,x2,y2 in lines[0]:
#     cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
# cv2.imshow('image', img)
# cv2.waitKey()

# img = cv2.imread('Documentatie/Puzzels/Tiles_scrambled/tiles_scrambled_2x2_00.png')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray,50,150,apertureSize = 3)
#
# lines = cv2.HoughLines(edges,1,np.pi/180,200)
# for rho,theta in lines[0]:
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))
#
#     cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
# cv2.imshow('image', img)
# cv2.waitKey()

# # Load image, convert to grayscale, Otsu's threshold for binary image
# image = cv2.imread('Documentatie/Puzzels/Tiles_scrambled/tiles_scrambled_2x2_00.png')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
#
# # Find contours, find rotated rectangle, obtain four verticies, and draw
# cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# rect = cv2.minAreaRect(cnts[0])
# box = np.int0(cv2.boxPoints(rect))
# cv2.drawContours(image, [box], 0, (36,255,12), 3) # OR
# # cv2.polylines(image, [box], True, (36,255,12), 3)
#
# cv2.imshow('image', image)
# cv2.waitKey()

3. # rect = cv2.minAreaRect(cnt)
# angle = rect[2]
# if angle < -45:
#     angle = (90 + angle)
# # otherwise, just take the inverse of the angle to make
# # it positive
# else:
#     angle = -angle
# # rotate the image to deskew it
# (h, w) = pieces[0][0].shape[:2]
# center = (w // 2, h // 2)
# M = cv2.getRotationMatrix2D(center, angle, 1.0)
# rotated = cv2.warpAffine(pieces[0][0], M, (w, h),
#     flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

"""