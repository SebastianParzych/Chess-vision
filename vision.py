import numpy as np
import glob
import cv2
from matplotlib import pyplot as plt
import time

def rescale_frame(frame, percent):
	width = int(frame.shape[1] * percent/ 100)
	height = int(frame.shape[0] * percent/ 100)
	dim = (width, height)
	return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

cap = cv2.VideoCapture('videos/video.mkv')
list_of_templates = []
for img in glob.glob("images/*.jpg"):
	template= cv2.imread(img)
	list_of_templates.append(template)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)
#
# result = cv2.VideoWriter('videos/video_4.mp4',
#                          cv2.VideoWriter_fourcc(*'mp4v'),30,size)
rows = ['a','b','c','d','e','f','g','h']
board = []
for row in rows:
	temp = []
	for j in range (1,9):
		temp.append(row+str(j))
	board.append(temp)
print(board)

#filter to sharpen frames
filter_9 = np.array([[0, -1, 0], [-1,5, -1], [0, -1, 0]])
kernel = np.array((
		[0, 1, 0],
		[1, 5, 1],
		[0, 1, 0]), dtype="int")
def contours_classification(contours, img): # Classify all chess pieces and board
   # cv2.drawContours(bishop, [contours], 0, (0, 255, 0), 4)
	for contour in contours:
		if (cv2.arcLength(contour, True) > 900): # Board Contur
			x, y, w, h = cv2.boundingRect(contour)
			cv2.drawContours(img, [contour], 0, (255, 0, 127), 10)
			cv2.putText(img, "BOARD", (x-100, y+25), cv2.FONT_ITALIC, 1, (255, 255,255))
		if(cv2.arcLength(contour, True) <195 or cv2.arcLength(contour,True)>600): # Filter unwanted contours
			continue
		approx = cv2.approxPolyDP(contour,0.03 * cv2.arcLength(contour, True), True) #Approximation of contours to distinguish them better
		cv2.drawContours(img, [approx], 0, (0, 255, 0), 4)
		area = cv2.contourArea(contour)
		x, y, w, h = cv2.boundingRect(approx)
		rect_area = w * h
		extent = float(area) / rect_area # shape coefficient
		aspect_ratio = float(w) / h
		M=cv2.moments(contour) # Moment to count centre of shape
		try:
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
		except:
			cX= 0
			cY= 0
		if (cv2.arcLength(contour, True)<210 ):
			cv2.rectangle(img, (x, y), (x + w, y + h), (255, 127, 0), 2)
			cv2.putText(img, "Pawn", (x, y), cv2.FONT_ITALIC, 0.5, (0, 0, 0))
			cv2.circle(img, (cX, cY), 7, ( 0, 255, 0), -1)
		if ( len(approx)==5 and cv2.arcLength(contour, True)>230 and extent>0.7 ):
			cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
			cv2.putText(img, "Queen", (x, y), cv2.FONT_ITALIC, 0.5, (0, 0, 0))
			cv2.circle(img, (cX, cY), 7, ( 0, 255, 0), -1)
		if ( extent<0.65 ):
			cv2.rectangle(img, (x, y), (x + w, y + h), (255, 51, 153), 2)
			cv2.putText(img, "Knight", (x, y), cv2.FONT_ITALIC, 0.5, (0, 0, 0))
			cv2.circle(img, (cX, cY), 7, ( 0, 255, 0), -1)
		if ( aspect_ratio<0.9 and aspect_ratio>0.7 and extent<0.9 and len(approx)==6):
			cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
			cv2.putText(img, "Rook", (x, y), cv2.FONT_ITALIC, 0.5, (1, 1, 1))
			cv2.circle(img, (cX, cY), 7, ( 0, 255, 0), -1)
		if ( aspect_ratio<1.1 and aspect_ratio>0.98 and len(approx)==6):
			cv2.rectangle(img, (x, y), (x + w, y + h), (46, 43, 95), 2)
			cv2.putText(img, "King", (x, y), cv2.FONT_ITALIC, 0.5, (0, 0, 0))
			cv2.circle(img, (cX, cY), 7, ( 0, 255, 0), -1)
		if (len(approx)==7 and extent<0.95 and extent>0.65 ):
			cv2.rectangle(img, (x, y), (x + w, y + h), (1139, 0, 255), 2)
			cv2.putText(img, "Bishop", (x, y), cv2.FONT_ITALIC, 0.5, (0, 0, 0))
			cv2.circle(img, (cX, cY), 5, ( 0, 255, 0), -1)
		if(len(approx) == 6 and float(area)>3000  and float(area)<3500 ):
			cv2.rectangle(img, (x, y), (x + w, y + h), (1139, 0, 255), 2)
			cv2.putText(img, "Bishop", (x, y), cv2.FONT_ITALIC, 0.5, (0, 0, 0))
			cv2.circle(img, (cX, cY), 5, (0, 255, 0), -1)

	cv2.imshow('asdas 123', img)
def process_image(image):
	gray = cv2.cvtColor(rescaled, cv2.COLOR_BGR2GRAY)
	equal = cv2.equalizeHist(gray)
	sharpen_9 = cv2.filter2D(equal, -1, filter_9)
	return sharpen_9,gray

while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()
	if ret == True:
		rescaled= rescale_frame(frame, 60)
		prep_image,gray  = process_image(rescaled)
		thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,5,3)
		_,threshold_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
		img_morph1 = cv2.morphologyEx(threshold_otsu, cv2.MORPH_ERODE, np.ones((7,7), np.uint8))
		canny_otsu = cv2.Canny(img_morph1,55,30)
		canny_standard = cv2.Canny(thresh, 55, 30)
		w,h=gray.shape
		contours, hierarchy = cv2.findContours(img_morph1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		img_black = np.zeros((w, h,3), dtype="uint8")
		lines = cv2.HoughLines(canny_standard, 1, np.pi / 180, 300)
		#print(intersections)
		img_line = np.zeros((w, h, 3), dtype="uint8")
		contours_classification(contours, rescaled.copy())
		cv2.imshow('Chess asda', canny_otsu)
		cv2.imshow('Chess 123', canny_standard)
		if cv2.waitKey(1) and  0xFF == ord('q'):
			break
	else:
		break



cap.release()
cv2.destroyAllWindows()


