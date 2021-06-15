import numpy as np
import glob
import cv2
import chess_engine
from datetime import datetime
class Chess_vision:
	def __init__(self):
		self.engine = chess_engine.Engine()
		self.start_board = {}
		self.cap = cv2.VideoCapture('videos/video.mkv')
		self.board_image= cv2.imread("images/board.jpg")
		self.board_image = cv2.cvtColor(self.board_image, cv2.COLOR_BGR2GRAY)
		self.rows = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
		self.filter_9 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
		self.game_moves = []
		self.board = {}
		self.dict_points = {}
		self.board_detected = False
		self.motion = False
		self.permition = False
		self.static_reference = None
	def __rescale_frame(sel,frame, percent):
		width = int(frame.shape[1] * percent/ 100)
		height = int(frame.shape[0] * percent/ 100)
		dim = (width, height)
		return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)
	def squares(self,x,y,w,h,img): # Estimate center point of each chess pol
		self.board_detected = True
		h = 1/8 * h
		w = 1/8 * w
		for number,row in enumerate(self.rows):
			for j in range (0,8):
				cur_x = round(x+h/2+h*number)
				cur_y =round(y+w/2+j*h)
				self.dict_points[row+str(abs(-8+(j)))] = [cur_x, cur_y]
				self.start_board=self.board[row+str(abs(-8+(j)))]=' '
				cv2.circle(img, (cur_x, cur_y), 10, (0, 255, 0), -1)
				cv2.putText(img, row+str(abs(-8+(j))), (cur_x , cur_y ), cv2.FONT_ITALIC, 1, (0, 0, 255))
		cv2.imshow('Chess-Vision-Board', img)
	def switch_name (self,name):
		switch= {
			"Pawn":     'P',
			"Queen":    'Q',
			"King":     'K',
			"Knight":   'N',
			"Bishop":   'B',
			"Rook":     'R'
		}
		return switch[name]
	def __set_piece(self,name,x,y,w,h,cX,cY,color,img,piece_color):
		cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
		cv2.putText(img, piece_color+" "+name, (x, y), cv2.FONT_ITALIC, 0.5, (0, 0, 0))
		cv2.circle(img, (cX, cY), 7, (0, 255, 0), -1)
		if dict == None or not self.permition:
			return
		for square,coordinate in self.dict_points.items():
			distance = pow(pow((coordinate[0]-cX),2)+pow((coordinate[1]-cY),2),0.5)
			if(distance<10):
				#if self.board[square] != ' ':
				self.board[square]=piece_color+"_"+self.switch_name(name)
	def check_empty(self,list_of_cord):
			place= False
			for square, coordinate in self.dict_points.items():
				for element in list_of_cord:
					distance = pow(pow((coordinate[0] - element[0]), 2) + pow((coordinate[1] - element[1]), 2), 0.5)
					if distance<10:
						place = True

				if place == False:
					self.board[square] =" "
			#print(self.board)
	def video_size(self):
		frame_width = int(self.cap.get(3))
		frame_height = int(self.cap.get(4))
		self.size = (frame_width, frame_height)
	def __contours_classification(self,contours, img,thresh): # Classify all chess pieces and board
		list_of_cord= []
		for contour in contours:
			if (cv2.arcLength(contour, True) > 900): # Board Contur
				x, y, w, h = cv2.boundingRect(contour)
				cv2.drawContours(img, [contour], 0, (255, 0, 127), 10)
				cv2.drawContours(thresh, [contour], 0, (255, 0, 127), 10)
				cv2.putText(img, "BOARD", (x-100, y+25), cv2.FONT_ITALIC, 1, (255, 255,255))
				if self.board_detected == True:
					continue
				self.squares(x, y, w, h, img.copy())
			if(cv2.arcLength(contour, True) <195 or cv2.arcLength(contour,True)>600): # Filter unwanted contours
				continue
			piece_color = "W"
			approx = cv2.approxPolyDP(contour,0.03 * cv2.arcLength(contour, True), True) #Approximation of contours to distinguish them better
			#cv2.drawContours(img, [approx], 0, (0, 0, 0), 4)
			area = cv2.contourArea(contour)
			x, y, w, h = cv2.boundingRect(approx)
			rect_area = w * h
			extent = float(area) / rect_area # shape factor
			aspect_ratio = float(w) / h
			M=cv2.moments(contour) # Moment to count centre of shape
			try:
				cX = int(M["m10"] / M["m00"])
				cY = int(M["m01"] / M["m00"])
			except:
				cX= 0
				cY= 0
			list_of_cord.append([cX,cY])
			colors = np.array(cv2.mean(thresh[y:y + h, x:x + w]))
			if (cv2.arcLength(contour, True)<210 ):
				if colors[0] > 60:
					piece_color = "B"
				self.__set_piece("Pawn", x, y, w, h, cX, cY, (0,255,255), img,piece_color  )
			if ( len(approx)==5 and cv2.arcLength(contour, True)>230 and extent>0.7 ):
				if colors[0] > 100:
					piece_color = "B"
				self.__set_piece("Queen", x, y, w, h, cX, cY, (255, 255, 0), img,piece_color)
			if ( extent<0.65 ):
				if colors[0] > 90:
					piece_color = "B"
				self.__set_piece("Bishop", x, y, w, h, cX, cY, (255, 51, 153), img,piece_color)
			if ( aspect_ratio<0.9 and aspect_ratio>0.7 and extent<0.9 and len(approx)==6):
				if colors[0] > 100:
					piece_color = "B"
				self.__set_piece("Rook", x, y, w, h, cX, cY, (0, 0, 255), img,piece_color)
			if ( aspect_ratio<1.1 and aspect_ratio>0.98 and len(approx)==6):
				if colors[0] > 100:
					piece_color = "B"
				self.__set_piece("King", x, y, w, h, cX, cY, (46, 150, 255), img,piece_color)
			if ((len(approx)==7 and extent<0.95 and extent>0.65 ) or (len(approx) == 6 and float(area)>3000  and float(area)<3500 )):
				if colors[0] > 90:
					piece_color = "B"
				self.__set_piece("Knight", x, y, w, h, cX, cY, (1139, 0, 255), img,piece_color)
			self.check_empty(list_of_cord)

			#if cv2.contourArea(contour) > 800:  # filter small contours
			#x, y, w, h = cv2.boundingRect(contour)  # offsets - with this you get 'mask'
			#cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
			#cv2.imshow('cutted contour', img[y:y + h, x:x + w])

			#print(mean)
		cv2.imshow('Chess-Vision', img)
	def process_image(self,image):
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		equal = cv2.equalizeHist(gray)
		sharpen_9 = cv2.filter2D(equal, -1, self.filter_9)
		return sharpen_9,gray
	def start(self):
		_,frame=self.cap.read()
		img =self.__rescale_frame(frame, 60)
		prep_image, gray = self.process_image(img)
		new_motion = False
		iterations =0
		while(True):
			iterations=iterations+1
			if(iterations==2):
				back_motion = new_motion
				frame_back = gray
				iterations=0
			ret, frame = self.cap.read()
			if ret:
				img= self.__rescale_frame(frame, 60)
				prep_image,gray  = self.process_image(img)
				thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,11,3)
				_,threshold_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
				img_morph1 = cv2.morphologyEx(threshold_otsu, cv2.MORPH_ERODE, np.ones((7,7), np.uint8))
				#canny_otsu = cv2.Canny(img_morph1,55,30)
				#canny_standard = cv2.Canny(gray, 33, 11)
				#w,h=gray.shape
				try:
				#	print(np.sum(np.absolute(frame_back - gray)) / np.size(gray))
					if np.sum(np.absolute(frame_back - gray)) / np.size(gray) > 1.05: #Register Motion
						self.motion=True
						new_motion= True
					else:
						self.motion=False
						new_motion = False
				except:
					pass
				try:
					if   back_motion and not new_motion: # Give permition to update virtual board only at end of motion
						self.permition =True
					else:
						if(self.permition==True):
							contours, hierarchy = cv2.findContours(img_morph1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
							self.__contours_classification(contours, img.copy(), thresh.copy())
							self.engine.vision_output(self.board)
						self.permition =False
				except:
					pass
				#cv2.imshow('Chess asda', img)
				if cv2.waitKey(1) and  0xFF == ord('q'):
					break
			else:
				break
		self.engine.game_evaluation_plot()
		self.cap.release()
		cv2.destroyAllWindows()


vision = Chess_vision()
vision.start()

