import cv2

# 웹캠에서 비디오 캡처 시작
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

# 웹캠에서 한 프레임을 읽어옵니다.
ret, frame = cap.read()

if ret:
    # 프레임이 정상적으로 읽혔다면, 이미지를 저장합니다.
    cv2.imwrite('/home/erp-42/catkin_ws/src/Advanced-Lane-Lines/camera_cal/captured_image20.jpg', frame)
    print("이미지가 저장되었습니다.")
else:
    # 프레임 읽기에 실패했다면, 메시지를 출력합니다.
    print("웹캠에서 이미지를 읽어오는 데 실패했습니다.")

# 캡처
