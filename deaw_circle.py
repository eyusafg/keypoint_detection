import cv2



img = cv2.imread(r'Datasets\cut_out\pressure_roi\img\masks_cloth\_20250917163425188.png')

cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
cv2.circle(img, (2000, 1650), 5, 255, -1)
cv2.imwrite('circle.png', img)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
