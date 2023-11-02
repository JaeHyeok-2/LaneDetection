## Line Detection Tutorials"

from util import * 

if __name__ == "__main__":
    input_type = input()

    if input_type =="image":
        frame = cv2.imread("datasets/solidWhiteCurve.jpg")
        if frame.shape[0] != 540: 
            frame= cv2.resize(frame, None, fx= 3/4, fy =3/4, interpolation=cv2.INTER_AREA)
        
        result = detect_lanes_img(frame) 

        cv2.imshow("Result", result)
        cv2.waitKey(0)

    
    elif input_type=="video":
        cap = cv2.VideoCapture("datasets/solidWhiteRight.mp4")
        while(cap.isOpened()):
            ret, frame = cap.read()
            if frame.shape[0] != 540: 
                frame = cv2.resize(frame, None, fx=3/4, fy=3/4,interpolation=cv2.INTER_AREA) 
            result = detect_lanes_img(frame) 
            
            cv2.imshow("result",result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break 


        cap.release()
        cv2.destroyAllWindows()

        