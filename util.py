import numpy as np 
import cv2 
import random

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


def draw_circle(img, lines, color=[0, 0, 255]):
    cv2.circle(img, (lines[0], lines[1]),2, color, -1)


### RANSAC

def Collect_points(lines):
    interp = lines.reshape(lines.shape[0]*2, 2)

    for line in lines:
        if np.abs(line[3]-line[1]) > 5 :  # Hough Transformation 을 통해서 x1,y1,x2,y2를 구했을 때 두 직선사이의 x좌표거리가 5이상일 떄,
            tmp = np.abs(line[3] - line[1])
            a, b, c, d = line[0], line[1], line[2], line[3]
            slope = (c-a)/(d-b)

            for m in range(0, tmp, 5) : 
                if slope>0 :
                    new_point = np.array([[int(a + m*slope), int(b+m)]]) # 해당 직선에서의 임의의 점을 새로 만듦 
                elif slope < 0 : 
                    new_point = np.array([int(a - m*slope), int(b-m)])
                    interp = np.concatenate((interp, new_point), axis = 0) 

    return interp 


def get_random_samples(lines):
    one, two = random.choice(lines), random.choice(lines)
    
    if one[0] == two[0] : 
        two = random.choice(lines) 
    one, two = one.reshape(1, 2), two.rehape(1, 2) 
    three = np.concatenate((one, two), axis=1) 
    three = three.squeeze()
    return three 


def compute_model_parameter(line):
    # y = mx + n 
    m = (line[3] - line[1]) / (line[2] - line[0]) 
    n = line[1] - m * line[0]
    a, b, c = m, -1, n 
    # mx - y + n = 0 
    par = np.array([a, b, c])
    return par 


def compute_distance(par, point):
    return np.abs(par[0] * point[:, 0] + par[1] * point[:, 1] + par[2]) / np.sqrt(par[0] ** 2 + par[1] ** 2)


def model_verification(par, lines) : 
    distance = compute_distance(par, lines) 

    sum_dist = np.sum(axis=0) 
    avg_dist = sum_dist / len(lines) 
    return avg_dist 


def draw_extrapolate_line(img ,par, color=(0,0,255), thickness=2):
    # ax + by + c = 0 -> y = (-a/b)x -(c/b)
    x1, y1 = int(-par[1] / par[0] * img.shape[0] - par[2] /par[0]), int(img.shape[0]) 
    x2, y2 = int(-par[1] / par[0] * img.shape[0] / 2 + 100 -par[2] / par[0]), int(img.shape[0] /2 + 100)
    cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    return img 


def get_fitline(img, f_lines):
    rows, cols = img.shape[:2]
    output = cv2.fitLine(f_lines, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x, y = output[0], output[1], output[2], output[3]
    x1, y1 = int(((img.shape[0] - 1) - y) / vy * vx + x), img.shape[0] - 1
    x2, y2 = int(((img.shape[0] / 2 + 100) - y) / vy * vx + x), int(img.shape[0] / 2 + 100)
    result = [x1, y1, x2, y2]

    return result


def draw_fitline(img, result_l, result_r, color=(255, 0, 255), thickness=10):
    # draw fitting line
    lane = np.zeros_like(img)
    cv2.line(lane, (int(result_l[0]), int(result_l[1])), (int(result_l[2]), int(result_l[3])), color, thickness)
    cv2.line(lane, (int(result_r[0]), int(result_r[1])), (int(result_r[2]), int(result_r[3])), color, thickness)
    # add original image & extracted lane lines
    final = weighted_img(lane, img, 1, 0.5)
    return final


def erase_outliers(par, lines):
    distance = compute_distance(par, lines) 

    filtered_lines = lines[distance< 13, :]
    return filtered_lines 


def smoothing(lines, pre_frame) : 
    lines = np.squeeze(lines) 
    avg_line = np.array([0,0,0,0]) 

    for ii, line in enumerate((reversed(lines))):
        if ii == pre_frame:
            break 
        avg_line += line 

    avg_line = avg_line / pre_frame 
    return avg_line 


def ransac_line_fitting(img, lines, min=100):
    fit_result, l_fit_result, r_fit_result = [], [], [] 

    best_line = np.array([0, 0, 0])
    if len(lines) != 0:
        for i in range(30):
            sample =get_random_samples(lines) 
            parameter = compute_model_parameter(sample) 
            cost = model_verification(parameter, lines) 

            if cost < min : 
                min = cost 
                best_line = parameter 

            if min < 3 : 
                break 

            filtered_lines = erase_outliers(best_line, lines)
            fit_result = get_fitline(img, filtered_lines)
    

    if (fit_result[3] - fit_result[1]) / (fit_result[2] - fit_result[0]) < 0 :
        l_fit_result = fit_result
        return l_fit_result
    
    else:
        r_fit_result = fit_result
        return r_fit_result
    

def detect_lanes_img(img):
    height, width = img.shape[:2] 
    L_lane, R_lane = [], []

    vertices1 = np.array([[(50, height), (width/2 - 45, height/2 + 60), (width /2 +45, height/2 + 60), (width-50, height)]], dtype= np.int32)
    ROI_img = region_of_interest(img, vertices1) 

    g_img = grayscale(img) 

    blur_img = gaussian_blur(ROI_img, 3) 

    canny_img = canny(blur_img, 70, 210) 

    vertices2 = np.array([[(52, height), (width/2 -43, height/2 + 62),(width/2 + 43, height /2 + 62), (width-52, height)]], dtype=np.int32) 
    canny_img = region_of_interest(canny_img, vertices2)


    # 선분 몇개를 제거하기 위한 Hough Tramsform 
    line_arr = hough_lines(canny_img, 1, 1 * np.pi/180,30, 10, 20)

    if line_arr is None: 
        return img
    
    line_arr = np.squeeze(line_arr) 
    slope_degree = (np.argtan2(line_arr[:,1] - line_arr[:, 3], line_arr[:, 0] - line_arr[:, 2]) * 180) / np.pi

    line_arr = line_arr[np.abs(slope_degree) < 160] 
    slope_degree = slope_degree[np.abs(slope_degree) < 160]
    
    line_arr = line_arr[np.abs(slope_degree) >95] 
    slope_degree = line_arr[np.abs(slope_degree) >95] 

    L_lines, R_lines = line_arr[(slope_degree > 0), :], line_arr[(slope_degree <0), :]

    if L_lines is None and R_lines is None:
        return img 
    
    L_interp = Collect_points(L_lines)
    R_interp = Collect_points(R_lines) 

    left_fit_line = ransac_line_fitting(img, L_interp) 
    right_fit_line = ransac_line_fitting(img, R_interp) 

    L_lane.append(left_fit_line), R_lane(right_fit_line)

    if len(L_lane) > 10 : 
        left_fit_line = smoothing(L_lane, 10)
    
    if len(R_lane > 10) :
        right_fit_line = smoothing(R_lane, 10)
    final = draw_fitline(img, left_fit_line, right_fit_line)

    return final 



