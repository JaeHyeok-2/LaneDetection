```The goals / steps of this project are the following  ```

- 주어진 체스보드 이미지셋에서 Calibration Matrix / distortion coefficient를 계산 
- 원본 이미지에 대해서 Distortion Correction을 적용 
- 이미지에 대해서 Color Transform, Graident를 적용 `&rarr` Threshold Binary image를 만들기 위함
- BEV와 같은 시점변화를 적용 
- lane pixel를 탐지하고, 차선 경계를 찾도록 모델을 fitting 
- 중심으로부터 자동차의 위치와 차선의 커브량을 결정 
- 자동차 위치와 차선의 변화량을 Estimation & 차선 경계를 시각화 




## Camera Calibraiton 

`실생활에 3D 객체를 카메라로 촬영을 하고, 이를 2D Image로 변형시켰을 때, 완벽하지 않은데, 이는 왜곡으로 인한 부작용입니다.`
`이러한 왜곡은 객체의 모양 변화, 차선 휘어짐 현상과 같은 왜곡된 정보를 제공하기 때문에 이를 해결하기 위해서 Camera Calibration 기법을 사용합니다.`
