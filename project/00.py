import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# %matplotlib inline

# img = cv2.imread( "C:\data\p_project/handpalm.jpeg",0)
# ret, tresh = cv2.threshold(img,127,255,0)
# contours, hierarchy = cv2.findContours(tresh, 1,2) 

# cnt = contours[30]
# M = cv2.moments(cnt)
# print(M)

img = mpimg.imread("C:\data\p_project/2.jpg")

# plt.figure(figsize=(10,8))
# print("This image is: ", type(img), "with dimensions:",img.shape)
# plt.imshow(img)
# plt.show()

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

gray = grayscale(img)
# plt.figure(figsize=(10,8))
# plt.imshow(gray, cmap='gray')
# plt.show()

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img,(kernel_size, kernel_size),0)

kernel_size = 5
blur_gray = gaussian_blur(gray, kernel_size)

# plt.figure(figsize=(10,8))
# plt.imshow(blur_gray, cmap='gray')
# plt.show()

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

low_threshold = 100
high_threshold = 20
edges = canny(blur_gray, low_threshold, high_threshold)

plt.figure(figsize=(10,8))
plt.imshow(edges, cmap='gray')
plt.show()

mask = np.zeros_like(img)

plt.figure(figsize=(10,8))
plt.imshow(mask, cmap='gray')
plt.show()

if len(img.shape) > 2:
    channel_count = img.shape[2]
    ignore_mask_color = (255,)
else:
    ignore_mask_color = 255

imshape = img.shape
print(imshape)

vertices = np.array([[(280, 300),
                      (280, 425),
                      (400, 425),
                      (400, 300)]], dtype=np.int32)

cv2.fillPoly(mask, vertices, ignore_mask_color)

plt.figure(figsize=(10,8))
plt.imshow(mask, cmap='gray')
plt.show()

def region_of_interest(img, vertices):
    #빈공간 정의
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count

    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

    #함수로정의
imshape = img.shape
vertices = np.array([[(280, 300),
                      (280, 425),
                      (400, 425),
                      (400, 300)]], dtype=np.int32)
mask = region_of_interest(edges, vertices)

plt.figure(figsize=(10,8))
plt.imshow(mask, cmap='gray')
plt.show()

def draw_lines(img, lines, color=[255,0,0], thickness =5):
    for line in lines :
        for x1, y1, x2,y2 in line:
            cv2.line(img,(x1, y1),(x2,y2), color, thickness)
# def hough_lines(img, rho, theta, threshold, min_line_len, miax_line_gap):
#     lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
#                 minLineLength = min_line_len,
#                 maxLineGap = max_line_gap)
#     line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
#     draw_lines(line_img, lines)
#     return line_img
# rho = 2
# theta = np.pi/180
# threshold = 90
# min_line_len = 120
# max_line_gap = 150

# lines = hough_lines(mask, rho, theta, threshold, min_line_len, max_line_gap)

# plt.figure(figsize=(10,8))
# plt.imshow(lines, cmap='gray')
# plt.show()    

# def weighted_img(img, initial_img, α=1, β=1., λ=0.):
#     return cv2.addWeighted(initial_img, α, img, β, λ)

# lines_edges = weighted_img(lines, img, α=0.8, β=1., λ=0.)

# plt.figure(figsize = (10,8))
# plt.imshow(lines_edges)
# plt.show()
