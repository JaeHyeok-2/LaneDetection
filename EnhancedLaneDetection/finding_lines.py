import numpy as np 
import cv2 
from PIL import Image 
import matplotlib.image as mpimg 


class Line: 
    def __init__(self) -> None:
        # 직전의 Iteration에서 line이 검출되었는가?
        self.detected = False 
        # windows의 마진 크기 
        self.window_margin = 56
        # n번째 iteration에서 fitted line의 값
        self.prevx = []
        # 최근 fit된 다항함수의 계수들
        self.current_fit = [np.array(False)]
        # 회전 반지름
        self.radius_of_curvature = None
        # x_value 시작값
        self.startx = None
        # x_value 끝값
        self.endx = None
        # 탐지한 line pixels의 x값
        self.allx = None 
        # 탐지한 line pixels의 y값
        self.ally = None 

        # Road Information
        self.road_inf = None
        self.curvature = None 
        self.deviation = None 



def wrap_image(img, src, dst, size):
    M = cv2.getPerspectiveTransform(src, dst) # return Transformation Matrix 
    Minv = cv2.getPerspectiveTransform(dst, src) 
    warp_img = cv2.warpPerspective(img, M, size, flags= cv2.INTER_LINEAR) 

    return warp_img, M, Minv 


def rad_of_curvature(left_line, right_line):
    """ measure radius of curvature """

    ploty = left_line.ally 
    leftx, rightx = left_line.allx, right_line.allx 

    leftx = leftx[::-1] 
    rightx = rightx[::-1] 

    # pixel x,y를 Meter단위로 바꾸자 -> 이미지의 크기 720 x1080 
    width_lanes = abs(right_line.startx - left_line.starty)
    ym_per_pix = 30 / 720 # 픽셀당 30 미터
    xm_per_pix = 3.7*(720/1080) /width_lanes # 종횡비 고려
    
    # y-value에 대한 radius of curvature(곡률)
    y_eval = np.max(ploty) 

    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx* xm_per_pix, 2)
    right_fit_cr = np.ployfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
    
    """
    곡률 = 1/ 곡선의 반지름
    
    곡선의 반지름(Radius of Curvation) =  (1 + 2A_yB_y + B_y **2)^(1.5)  / abs(2 * A_y) 
    A_y : 2차 다항식의 2차 항의 계수
    B_y : 2차 다항싕 1차 항의 계수
    
    """
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 *right_fit_cr[1]) 
    left_line.radius_of_curvature = left_curverad
    right_line.radius_of_curvature = right_curverad 


def smoothing(lines, pre_lines=3):
    # collect lines & print average line 
    lines = np.squeeze(lines) 
    avg_line = np.zeros((720)) 

    for ii, line in enumerate(reversed(lines)):
        if ii == pre_lines:
            break 
        avg_line += line 
    avg_line = avg_line / pre_lines 

    return avg_line 



def blind_search(b_img, left_line, right_line):
    """
    blind_search  : 첫번째 프레임 혹은, 차선을 잃어버렸을 때,
    Histogram & sliding Window를 통해서 찾아냄
    
    """
    # GrayScale의 이미지로 입력이 들어올것이고, 배경은 제외하는 범위 
    histogram = np.sum(b_img[int(b_img.shape[0] / 2):, :], axis = 0) # (960, 1)

    output =np.dstack((b_img, b_img, b_img)) * 255 # 이 작업은 잘 이해가 되지않음..(960,1)을 depth방향을 stack을 쌓으면 뭐하는거지 어짜피 GrayScale이라 모두 값이 동일하지 않나

    midpoint = np.int(histogram.shape[0] / 2)
    start_leftX = np.argmax(histogram[:midpoint])
    start_rightX = np.argmax(histogram[midpoint:]) + midpoint 

    num_windows = 9 

    window_height = np.int(b_img.shape[0] / num_windows)
    
    nonzero = b_img.nonzero()  # H W C 순임
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])
    
    current_leftX = start_leftX 
    current_rightX = start_rightX 

    min_num_pixel = 50 

    win_left_lane, win_right_lane = [], []

    window_margin= left_line.window_margin 


    for window in range(num_windows):

        win_y_low = b_img.shape[0] - (window + 1) * window_height
        win_y_high = b_img.shape[1] - window * window_height 
        
        # current X는 각각의 차선 중간 지점이라고 생각하면 됌 
        win_leftx_min = current_leftX - window_margin 
        win_leftx_max = current_leftX + window_margin

        win_rightx_min = current_rightX - window_margin
        win_rightx_max = current_rightX + window_margin

        cv2.rectangle(output, (win_leftx_min,win_y_low), (win_rightx_max, win_y_high), (0, 255, 0), 2) 
        cv2.rectangle(output, (win_rightx_max, win_y_low), (win_rightx_max, win_y_high), (0, 255, 0), 2)


        # 각 윈도우의 내부에 있는 coordinates를 저장. 이후에 중심점 조정할때 쓰일 수 있음
        left_win_inds = ((nonzero_y >= win_y_low) & (nonzero_y <=win_y_high) & (nonzero_x >= win_leftx_min) & (nonzero_x <=win_leftx_max))
        right_win_inds = ((nonzero_y >= win_y_low) & (nonzero_y <=win_y_high) & (nonzero_x >= win_rightx_min) & (nonzero_x <= win_rightx_max))

        win_left_lane.append(left_win_inds)
        win_right_lane.append(right_win_inds) 


        leftx, lefty = nonzero_x[win_left_lane], nonzero_y[win_left_lane]
        rightx, righty = nonzero_x[win_right_lane], nonzero_y[win_right_lane]

        output[lefty, leftx] = [255, 0, 0]
        output[righty, rightx] = [0, 0, 255] 


        left_fit =np.polyfit(lefty, leftx, 2) 
        right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, b_img.shape[0]-1, b_img.shape[0]) 

        # ax^2 + bx + c 
        left_plotx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2] 
        right_plotx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2] 
        

        left_line.prevx.append(left_plotx) 
        right_line.prevx.append(right_plotx)


        if len(left_line.prevx) > 10 : 
            left_avg_line = smoothing(left_line.prevx, 10)
            left_avg_fit = np.polyfit(ploty, left_avg_line, 2)
            left_fit_plotx = left_avg_fit[0] * ploty ** 2 + left_avg_fit[1] * ploty + left_avg_fit[2]
            left_line.current_fit = left_avg_fit 
            left_line.allx, left_line.ally = left_fit_plotx, ploty 
        
        else : 
            left_line.cuurent_fit = left_fit 
            left_line.allx, left_line.ally = left_fit_plotx, ploty 


        if len(right_line.prevx) > 10:
            right_avg_line = smoothing(right_line.prevx, 10)
            right_avg_fit = np.polyfit(ploty, right_avg_line, 2)
            right_fit_plotx = right_avg_fit[0] * ploty ** 2 + right_avg_fit[1] * ploty + right_avg_fit[2]
            right_line.current_fit = right_avg_fit
            right_line.allx, right_line.ally = right_fit_plotx, ploty
        else:
            right_line.current_fit = right_fit
            right_line.allx, right_line.ally = right_plotx, ploty

        # height 가 0 -> 540으로 탐색을 시작하니까, 맨위에 있는 값이 배열의 맨 마지막 값으로 들어가기 때문에 
        left_line.startx, right_line.startx = left_line.allx[len(left_line.allx) - 1], right_line.allx[len(right_line.allx) - 1] 
        left_line.endx, right_line.endx = left_line.allx[0], right_line.allx[0] 

        left_line.detected, right_line.detected = True, True 

        rad_of_curvature(left_line, right_line)
        return output
    



def prev_window_refer(b_img, left_line, right_line):
    """
    refer to previous window info - after detecting lane lines in previous frame
    """
    # Create an output image to draw on and  visualize the result
    output = np.dstack((b_img, b_img, b_img)) * 255

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = b_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Set margin of windows
    window_margin = left_line.window_margin

    left_line_fit = left_line.current_fit
    right_line_fit = right_line.current_fit
    leftx_min = left_line_fit[0] * nonzeroy ** 2 + left_line_fit[1] * nonzeroy + left_line_fit[2] - window_margin
    leftx_max = left_line_fit[0] * nonzeroy ** 2 + left_line_fit[1] * nonzeroy + left_line_fit[2] + window_margin
    rightx_min = right_line_fit[0] * nonzeroy ** 2 + right_line_fit[1] * nonzeroy + right_line_fit[2] - window_margin
    rightx_max = right_line_fit[0] * nonzeroy ** 2 + right_line_fit[1] * nonzeroy + right_line_fit[2] + window_margin

    # Identify the nonzero pixels in x and y within the window
    left_inds = ((nonzerox >= leftx_min) & (nonzerox <= leftx_max)).nonzero()[0]
    right_inds = ((nonzerox >= rightx_min) & (nonzerox <= rightx_max)).nonzero()[0]

    # Extract left and right line pixel positions
    leftx, lefty = nonzerox[left_inds], nonzeroy[left_inds]
    rightx, righty = nonzerox[right_inds], nonzeroy[right_inds]

    output[lefty, leftx] = [255, 0, 0]
    output[righty, rightx] = [0, 0, 255]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, b_img.shape[0] - 1, b_img.shape[0])

    # ax^2 + bx + c
    left_plotx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_plotx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    leftx_avg = np.average(left_plotx)
    rightx_avg = np.average(right_plotx)

    left_line.prevx.append(left_plotx)
    right_line.prevx.append(right_plotx)

    if len(left_line.prevx) > 10:
        left_avg_line = smoothing(left_line.prevx, 10)
        left_avg_fit = np.polyfit(ploty, left_avg_line, 2)
        left_fit_plotx = left_avg_fit[0] * ploty ** 2 + left_avg_fit[1] * ploty + left_avg_fit[2]
        left_line.current_fit = left_avg_fit
        left_line.allx, left_line.ally = left_fit_plotx, ploty
    else:
        left_line.current_fit = left_fit
        left_line.allx, left_line.ally = left_plotx, ploty

    if len(right_line.prevx) > 10:
        right_avg_line = smoothing(right_line.prevx, 10)
        right_avg_fit = np.polyfit(ploty, right_avg_line, 2)
        right_fit_plotx = right_avg_fit[0] * ploty ** 2 + right_avg_fit[1] * ploty + right_avg_fit[2]
        right_line.current_fit = right_avg_fit
        right_line.allx, right_line.ally = right_fit_plotx, ploty
    else:
        right_line.current_fit = right_fit
        right_line.allx, right_line.ally = right_plotx, ploty

    # goto blind_search if the standard value of lane lines is high.
    standard = np.std(right_line.allx - left_line.allx)

    if (standard > 80):
        left_line.detected = False

    left_line.startx, right_line.startx = left_line.allx[len(left_line.allx) - 1], right_line.allx[len(right_line.allx) - 1]
    left_line.endx, right_line.endx = left_line.allx[0], right_line.allx[0]

    # print radius of curvature
    rad_of_curvature(left_line, right_line)
    return output



def find_LR_lines(binary_img, left_line, right_line):
    """
    find left, right lines & isolate left, right lines
    blind search - first frame, lost lane lines
    previous window - after detecting lane lines in previous frame
    """

    if left_line.detected == False:
        return blind_search(binary_img, left_line, right_line)
    else:
        return prev_window_refer(binary_img, left_line, right_line)


def draw_lane(img, left_line, right_line, lane_color=(255, 0, 255), road_color= (0, 255, 0)):
    """ draw lane lines & current driving space """
    window_img = np.zeros_like(img) 

    window_margin = left_line.window_margin
    left_plotx, right_plotx = left_line.allx, right_line.allx
    ploty = left_line.ally


    left_pts_l = np.array([np.transpose(np.vstack([left_plotx - window_margin/5, ploty]))])
    left_pts_r = np.array([np.flipud(np.transpose(np.vstack([left_plotx + window_margin/5, ploty])))])
    left_pts = np.hstack((left_pts_l, left_pts_r)) 

    right_pts_l = np.array([np.transpose(np.vstack([right_plotx -window_margin/5, ploty]))]) 
    right_pts_r = np.array([np.filpud(np.transpose(np.vstack([right_plotx + window_margin/5, ploty])))])
    right_pts = np.hstack((right_pts_l, right_pts_r))

    cv2.fillPoly(window_img, np.int_([left_pts]), lane_color) 
    cv2.fillPoly(window_img, np.int_([right_pts]), lane_color) 

    pts_left = np.array([np.transpose(np.vstack([left_plotx+ window_margin/5, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_plotx - window_margin/5, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(window_img, np.int_([pts]), road_color)
    result = cv2.addWeighted(img, 1, window_img, 0.3, 0)
    return result, window_img 