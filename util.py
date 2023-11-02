import numpy as np 
import cv2 


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices, color3=(255, 255, 255), color1=255):
    # ROIs Settings 

    mask = np.zeros_like(img) #

    if len(img.shape) > 2: 
        color = color3  # color image 
    else : 
        color = color1 # gray image

    cv2.fillPoly(mask, vertices, color) 

    ROI_image = cv2.bitwise_and(img, mask) # 
    return ROI_image 


def draw_lines(img, lines, color=[0, 0, 255], thickness=2):
    for line in lines: 
        for x1, y1, x2, y2 in line : 
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)



def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):

    """
    image = 1채널의 흑백 이미지만을 입력으로 받음  보통 Canny를 통해 edge를 찾은 후에 이 함수를 적용하므로 이미 흑백으로 변환된 상태
    
    \rho : hough space에서 rho값을 하번에 얼만큼 증가시키면서 조사할 것인가?
    
    \theta : 단위는 라디안으로, 보통 각도를 입력한 후 pi/180을 곱해서 라디안 값으로 변환, [0:180]을default로 하는데, 180도를 넘긴 순간 부터 직선이 unique해지지 않기 때문에

    threshold : Hough Space에서 교차점이 있는데, Hough Transform에서는 이 교차점이 하나씩 증가할 때 마다 +1을 해줌
                Image Space관점에서 말하자면, 서로 일직선 위에 있는 점의 수가 threshold 개수 이상인지 아닌지를 판단하는 척도와 같은 말
                threshold값이 작으면 많은 직선이 검출 

    output : 검출된 직선 만큼의 rho, theta값 
    """
    # HoughLinesP는 확률적이라는 말이 포함 된건데 minLenght, maxLineGap은 각각 , 최소 선분의 길이를 추출 / 선 위의 점들 사이 최대 거리로 이 값보다크면 나와는 다른 선분으로 간주하겠다 의미
    # output으로는 선붕늬 시작점과 끝점에 대한 좌표 값
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), 
                            minLineLength = min_line_len,
                            maxLineGap=max_line_gap)
    
    line_img =np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # draw_lines(line_img, lines)
    
    return lines 

def weighted_img(img ,initial_img, a=1, b=1., lamb=0.):
    return cv2.addWeighted(initial_img, a, img, b, lamb) 


def get_fitline(img, f_lines):
    lines = np.squeeze(f_lines) 
    lines = lines.reshape(lines.shape[0]*2, 2)
    rows, cols = img.shape[:2] 

    output = cv2.fitLine(lines, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x, y = output[0], output[1], output[2], output[3] 

    vx, vy, x, y = output[0], output[1], output[2], output[3] 

    x1, y1 = int(((img.shape[0]-1)-y)/vy*vx + x), img.shape[0]-1
    x2, y2 = int(((img.shape[0]/2+100)-y)/vy*vx + x) , int(img.shape[0]/2+100)

    result = [x1,y1,x2,y2]
    return result 

def draw_fit_line(img, lines, color=[255, 0, 0], thickness=10) :
    cv2.line(img, (lines[0], lines[1]), (lines[2], lines[3]), color, thickness)