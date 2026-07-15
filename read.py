import cv2

cap = cv2.VideoCapture(0)
cap.set(3,500)
cap.set(4,500)

while True:
    success, img = cap.read()
    # img = 
    img2 = cv2.flip(img,1)
    print(img.shape)
    # cv2.putText(img,'Hello',(100,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    # cv2.circle(img,(235,235),50,(255,0,0),5)
    cv2.imshow('Affan',img2)

    cv2.waitKey(1)







# import cv2

# import numpy as np

# img = cv2.imread('images.jpeg')
# blank = np.zeros_like(img)
# blank[:] = 0,255,0
# combine = cv2.addWeighted(img,0.8,blank,0.2,0)

# imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# cv2.imshow('Combine',combine)

# cv2.imshow('Image',img)
# cv2.imshow('Blank',blank)
# cv2.waitKey(0)





