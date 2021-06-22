import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils


def img_equal_clahe_yuv(img):
    """Apply equalize, clahe into image with YUV channels.

    Args:
        img (image) : Image
    Returns:
        img_eq (image):  Image with equalize applied
        img_clahe (image): Image with clahe applied
    """
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    img_eq = img_yuv.copy()
    img_eq[:,:,0] = cv2.equalizeHist(img_eq[:, :, 0])
    img_eq = cv2.cvtColor(img_eq, cv2.COLOR_YUV2BGR)

    img_clahe = img_yuv.copy()
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)) #CLAHE 생성
    img_clahe[:, :, 0] = clahe.apply(img_clahe[:, :, 0])        #CLAHE 적용
    img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_YUV2BGR)
    return img_eq,img_clahe


def img_normalize(img):
    """Normalize image

    Args:
        img (image) : Image

    Returns:
        img_norm (image): Image with normalize applied
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_norm = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX)
    return img_norm


def show_hist(img):
    """Visual histogram of image

    Args:
        img (image) : Gray scale image
    """
    plt.hist(img.flatten(), 256, [0, 256], color = 'r')
    plt.xlim([0, 256])
    plt.show()
    plt.figure()


def img_clahe_luminus(img):
    """Apply equalize, clahe into image with LAB channels.

    Args:
        img (image) : Image
    Returns:
        img_eq (image):  Image with equalize applied
        img_clahe (image): Image with clahe applied
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # luminosity(명도)채널을 얻기 위해 채널을 BGR->LAB 로 바꿈
    l, a, b = cv2.split(lab) # 채널 분리

    el=cv2.equalizeHist(l)
    img_eq=cv2.merge((el, a, b))
    img_eq=cv2.cvtColor(img_eq, cv2.COLOR_LAB2BGR)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)) # 히스토그램 균등화 시키기 http://www.gisdeveloper.co.kr/?p=6652
    cl = clahe.apply(l) # 명도 채널에 적용
    limg = cv2.merge((cl, a, b)) # 채널 합치기
    img_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR) 
    return img_eq, img_clahe


def remove_line(gray):
    """Remove long line

    Args:
        gray (image): Gray scale image

    Returns:
        255 - result: Binary image with removed long line.
    """
    h,w = gray.shape[:2] # h, w
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  # gray 채널
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]  #OTSU 방법을 이용해 binary 이미지 변환

    kernel = np.ones((3, 3), np.uint8)
    dilation_image = cv2.dilate(thresh, kernel, iterations=1) # dilate 연산 

    # 가록 선을 찾을 커널 정하기 -> 가로 선을 찾을 꺼나 (x,y)에서 x를 더 크게 잡아야된다.
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1)) 
    # 모포로지 연산 (위의 dilate와 유사한 시리즈)
    detected_lines = cv2.morphologyEx(dilation_image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    # 선들의 contour 들을 찾기
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    temp_height = 10
    for c in cnts: # 모든 contour 마다
        temp = c.flatten()
        if max(temp[::2]) - min(temp[::2]) > w / 2: # contour의 w가 전체 이미지의 w/2보다 크면 삭제
            temp_height = max(temp[1::2]) - min(temp[1::2])
            cv2.drawContours(dilation_image, [c], -1, (0, 0, 0), -1) # 안을 검은색으로 채워줌

    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, (temp_height // 10) + 1))
    result = cv2.morphologyEx(dilation_image, cv2.MORPH_CLOSE, repair_kernel, iterations=2)
    return 255 - result


def global_threshold1(gray):
    """Binary image using global threshold

    Args:
         gray (image): Gray scale image

    Returns:
        gray (image) : Resize gray scale image (It has same size with returned binray image)
        255 - closing (image) : Binary image
    """
    h, w = gray.shape[:2] # h, w
    if w > 1000 and h > 100: # 가로 세로가 일정 수치 보다 크면 30%씩 줄임
        gray=cv2.resize(gray, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
    blurred = cv2.GaussianBlur(gray, (11, 11), 1) # 가우시안 블러 적용
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]  #OTSU 방법을 이용해 binary 이미지 변환
    kernel =  cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) # 사각형 모양의 커널을 만듬
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel) # 모포로지 open
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel) # 모포로지 close
    return gray, 255 - closing


def remove_brightness(gray):
    """Applt binary to image with severe brightness deviation.

    Args:
         gray (image): Gray scale image

    Returns:
        gray (image) : Resize gray scale image (It has same size with returned binray image)
        255 - closing (image) : Binary image
    """
    h, w = gray.shape[:2]
    if w > 1000 and h > 100: # 가로 세로가 일정 수치 보다 크면 30%씩 줄임
        gray=cv2.resize(gray, dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
    blurred = cv2.GaussianBlur(gray, (11, 11), 1)  # 가우시안 블러 적용
    thresh=cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # adaptiveThreshold를 이용해 binary 이미지 만들기
                                 cv2.THRESH_BINARY, 15, 2)

    kernel =  cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) # 사각형 모양의 커널을 만듬
    opening = cv2.morphologyEx(255-thresh, cv2.MORPH_OPEN, kernel) # 모포로지 open
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)) # 모포로지 close를 위한 커널
    result = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, repair_kernel, iterations=1) # 모포로지 close
    return gray, 255 - result
    

def show_x_y_hist(img):
    """Visual image's the image mean in the x,y axis.

    Args:
        img (image) : Image
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    temp = []
    for i in range(w):
        temp.append(gray[:, i].mean())
    plt.title('x')
    plt.plot(temp)
    plt.figure()
    temp = []
    for i in range(h):
        temp.append(gray[i, :].mean())
    plt.title('y')
    plt.plot(temp)
    plt.figure()
    plt.imshow(img)


def pyramid(image, scale=1.5, minSize=(30, 30)):
	# yield the original image
	yield image
	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)
		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
		# yield the next image in the pyramid
		yield image


def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def sliding_window1(gray):
    """Check the image brightness deviation using the sliding window method.

    Args:
        gray (image): Gray scale image

    Returns:
        _max (float) : maximum brightness of image.
        _min (float) : minimum brightness of image.
    """
    h, w = gray.shape[:2]
    (w_width, w_height) = (w // 3, h // 3) # sliding_window 크기
    dead_line = min(w // 10, h // 10) # sliding_window의 종료 지점
    _max = 0
    _min = np.inf
    for x in range(0, gray.shape[1] - dead_line, w // 5): # sliding window는 w//5의 크기 만큼 움직임
        for y in range(0, gray.shape[0] - dead_line, h // 5): # sliding window는 h//5의 크기 만큼 움직임
            t_x, t_y = x + w_width, y + w_height # sliding window의 다음 지점
            if x + w_width > gray.shape[1]:
                t_x = gray.shape[1]
            if y + w_height > gray.shape[0]:
                t_y = gray.shape[0]
            window = gray[x:t_x, y:t_y] # 현재 window
            _mean = window.mean() # 현재 window의 평균 밝기 값
            if _max <_mean:
                _max = _mean
            if _min > _mean:
                _min = _mean  
    return _max, _min


def sliding_window2(gray):
    """Check the image brightness deviation using the sliding window method.

    Args:
        gray (image): Gray scale image

    Returns:
        _max (float) : maximum brightness of image.
        _min (float) : minimum brightness of image.
    """
    h, w = gray.shape[:2]
    _max = 0
    _min = np.inf
    (winW, winH) = (w // 3, h // 3)
    for resized in pyramid(gray, scale=1.5):
        for (x, y, window) in sliding_window(resized, stepSize=max(w // 10, h // 10), windowSize=(winW, winH)):
            if window.shape[0] != winH or window.shape[1] != winW:
                continue
            _mean = resized[y:y + winH, x:x + winW].mean()
            if _max <_mean:
                _max = _mean
            if _min > _mean:
                _min = _mean
    return _max, _min