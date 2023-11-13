"""
Camera Calibration을 하기 위한 CheckBoard 한장의 이미지를 여러 방향에서 찍은 이미지들을 모아서, 카메라 내부파라미터 Rotation R, Translation T 를 조정하는 방식


"""
import numpy as np
import cv2 
import matplotlib.image as mpimg 
import glob 
import os 

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

images = glob.glob('datasets/cal/calibration*.jpg')
print(images)
obj_points = [] # 3D Points in real world space 
img_points = [] # 2D Points in image plane 

def calib():
    """
    왜곡되지 않은 이미지를 얻기 위해서, Camera Matrix & Distortion Coefficient가 필요함
    9*6의 20장 체스보드 이미지를 사용하여 계산
    """
    objp = np.zeros((6*9, 3), np.float32) 
    # np.mgrid[0:9, 0:6] 
    """
    objp[:,0] = [[0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],]
    """

    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
    
    for fname in images:
        img = mpimg.imread(fname) 
        gray =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # print(gray.shape)
        """
        cv2.findChessboardCorners(image, patternSize, flags) 
        
        args:
            image : 체커보드 사진, 8-bit grayscale or Color
            patternSize : 체커보드 행과 열당 내부 코너 개수
            corners : 감지된 코너의 출력 배열
            flags : 다양한 작업 플래그
            
            패턴이 감지되었는지 여부에 따라 출력은 T/F
        
        """
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None) 
        
        if ret== True: 
            obj_points.append(objp) # 체스보드 패턴의 코너위치 
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1, -1), criteria)
            img_points.append(corners) # 추출한 패턴의 위치

        else:
            print("해당 이미지는 탐지 실패!")
            continue 
    
    """
        obj_points : 3D 점 벡터로 구성된 벡터. 외부 벡터는 패턴 사진의 수만큼 요소를 포함
        img_points : 2D 이미지 점 벡터로 구성된 벡터
        Imagesize : 이미지의 크기
        distCoeffs :렌즈 왜곡 계수
        rvecs : Rotation Vector / 3x1 vector 로 벡터의 방향은 회전 축을 지정, 크니는 회전 각을 지정
        tvecs : 3x1 Translation Vector 
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    return mtx, dist 


def undistort(img, mtx, dist):
    """
    src : 왜곡된 입력 이미지
    cameraMatrix : 카메라의 내부 파라미터 행렬 
    disCoeffs: 왜곡 계수를 나타내는 배열로, 카메라 렌즈 왜곡을 정의 -> Camera calibration을 통해서 얻어낼 수 있음
    dst : 왜곡이 보정된 출력 이미지
    newCameraMAtrix : 보정된 이미지의 내부 파라미터 행렬 (Default :cameraMatrix와 동일 )
    """
    return cv2.undistort(img, mtx, dist, None, mtx) 


mtx, dist = calib() 
print(mtx, dist)