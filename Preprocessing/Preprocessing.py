import cv2
from PIL import Image,ImageEnhance

# Step1.Transform the image RGB into HLS models
#Hue(顏色)、Lightness(亮度)、Saturation(飽和度)
img = cv2.imread("test.jpg")
imgHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS) 

# Step2. Splitting the HLS image to different channels
h,l,s = hls_planes = cv2.split(imgHLS) # BGR # HLS

# H ==> Hue band


# S ==> Saturation ==>Histogram Equalization
S = cv2.equalizeHist(s)

# L ==> Lightness ==> 
# Histogram
histSize = 256
histRange = (0, 256) # the upper boundary is exclusive
g_hist = cv2.calcHist(hls_planes, [1], None, [histSize], histRange, accumulate=False)


def clip_computing(image, hist, low_clip, high_clip):
    total = image.shape[0]*image.shape[1]
    gray_L = -1
    gray_H = 256
    suml = sumh = 0.0    
    # find gray in low_clip
    while ( (suml/total) < low_clip):
        gray_L += 1
        suml += hist[gray_L]
    if ( (suml/total) >= (low_clip + 0.01)) :
        gray_L -= 1
    
    # find gray in high_clip
    while ( (sumh/total) < high_clip):
        gray_H -= 1
        sumh += hist[gray_H]
    if ( (sumh/total) >= (high_clip + 0.01)) :
        gray_H += 1
    
    return gray_L, gray_H



# 當灰階值image(i,j)<=gray_L => image(i,j) = gray_L+1
# 當灰階值image(i,j)>=gray_H => image(i,j) = gray_H-1
def stretch_image(image, gray_L, gray_H, low_limit, high_limit):
    rows = image.shape[0]
    cols = image.shape[1]
    # Modify image
    for i in range(rows):
        for j in range(cols):
            if image[i][j] <= gray_L: image[i][j] = gray_L+1
            if image[i][j] >= gray_H: image[i][j] = gray_H+1    
                
    # Stretch image
    delta_gray  = gray_H - gray_L
    delta_limit = high_limit - low_limit
    for i in range(rows):
        for j in range(cols):
            image[i][j] = (image[i][j]-gray_L)/delta_gray*delta_limit+low_limit
    
    return image
L= stretch_image()

# Step3 Combine processed HSL bands
limg = cv2.merge([h,L,S])
cv2.imshow('limg',limg)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Step 4 Conversion of HLS Color space to RGB
# Transform the image HLS into RGB models
imgRGB = cv2.cvtColor(limg,cv2.COLOR_HLS2RGB)
cv2.imshow('imgBGR',imgRGB)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("test2.jpg", imgRGB) # Preprocessing Save

# Step 5 Contrast Enhanced output image 
im = Image.open('test2.jpg','r')   

# 對比度增強
enh_con = ImageEnhance.Contrast(im)
contrast = 10 # 設定
image_contrasted = enh_con.enhance(contrast)
image_contrasted.show()
image_contrasted.save("test3.jpg")


