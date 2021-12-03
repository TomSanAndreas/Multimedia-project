#bron: https://stackoverflow.com/questions/55169645/square-detection-in-image
import cv2
import numpy as np
import sys

"""Source:  https://stackoverflow.com/questions/45613544/python-opencv-cannot-change-pixel-value-of-a-picture
            https://www.geeksforgeeks.org/python-grayscaling-of-images-using-opencv/
"""

# Figuur inlezen
image = cv2.imread('Documentatie/Puzzels/Tiles_scrambled/tiles_scrambled_2x2_00.png')

#figuur converteren naar grijswaarden
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#optioneel: lengte en breedte bepalen (nodig voor methode 2)
# (row,col) = gray.shape[0:2]
# print(row, col)

#array achter de figuur printen
#np.set_printoptions(threshold=sys.maxsize)
#print(gray)
#cv2.imshow('test1', gray)

#puzzeltegels witmaken vanuit grijswaardenbeeld (methode1)
notblack=np.where((gray[:,:]!=0))
gray[notblack]=(255)

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
cv2.imshow('test', gray)
# cv2.waitKey()
# cv2.destroyAllWindows()

canny = cv2.Canny(gray, 130, 255, 1)
# hough = cv2.HoughLines(canny, 1, np.pi / 180, 150, None, 0, 0)
cnts = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    cv2.drawContours(image,[c], 0, (0,255,0), 2)
cv2.imshow("result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


"""
Oude code (backup)
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
"""