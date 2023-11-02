"""
Shadow, Curve Line에 대해서는 어떻게 처리할 것인가?

"""
import os
from util import weighted_img, draw_lines, canny, gaussian_blur, grayscale, hough_lines,region_of_interest,get_fitline, draw_fit_line
import numpy as np 
import cv2

img = cv2.imread("datasets/slope_test.jpg")
cv2.imshow("image", img)

height, width = img.shape[:2]

gray_img = grayscale(img) 
blur_img =gaussian_blur(gray_img,3)
canny_img = canny(blur_img, 70, 210) 

vertices = np.array([[(50,height),(width/2-45, height/2+60), 
                      (width/2+45, height/2+60), (width-50,height)]], dtype=np.int32)
ROI_img = region_of_interest(canny_img, vertices) # ROI 설정



"""
내 차선에 평행한 직선들만 남기면 되므로, Hough Algorithm에서 직선의 기울기가 y축과 한없이 가까운 직선만 남기면 되잖아?

"""
line_arr =hough_lines(ROI_img, 1, 1 * np.pi/180, 30, 10, 20)
line_arr = np.squeeze(line_arr) 


slope_degree =  (np.arctan2(line_arr[:,1] - line_arr[:, 3], line_arr[:, 0] - line_arr[:, 2]) * 180) / np.pi 


# 수평 기울기 제한
line_arr = line_arr[np.abs(slope_degree) < 160] 
slope_degree = slope_degree[np.abs(slope_degree) < 160]

# 수직 기울기 제한 
line_arr = line_arr[np.abs(slope_degree)> 95]
slope_degree = slope_degree[np.abs(slope_degree) > 95]

# 필터링된 직선 버리기

L_lines, R_lines = line_arr[(slope_degree >0), :], line_arr[(slope_degree<0), :]
temp = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8) 

L_lines, R_lines = L_lines[:, None], R_lines[:,None]
print(L_lines.shape, R_lines.shape)

left_fit_line =get_fitline(img, L_lines)
right_fit_line = get_fitline(img, R_lines)

print(left_fit_line, right_fit_line)

# 대표선 그리기
draw_fit_line(temp, left_fit_line)
draw_fit_line(temp, right_fit_line)

cv2.imshow('TEMP', temp)
result = weighted_img(temp, img)
cv2.imshow('result', result) 
cv2.waitKey(0)