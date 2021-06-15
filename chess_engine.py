import matplotlib.pyplot as plt
import numpy as np

class Engine:
	def __init__(self):
		self.__match_evaluation = 0
		self.__game_evaluation_progres = [self.__match_evaluation]
		self.last_board ={}
		self.history = []

	def vision_output(self, board):
		board= board.copy()
		if not self.last_board:
			self.last_board = board.copy()
			return
		try:
			for key,value in board.items():                    # Recording only new moves
				if self.last_board[key]!=value and value !=' ':
					if(self.last_board[key][0] == value[0]): # Additional filltering vision output
						self.history.pop()
					self.history.append(value+key)
					sum_of_points = []
					for keys in board.keys():
						sum_of_points.append(self.__Piece_value(board[keys]))
					self.__game_evaluation_progres.append(sum(sum_of_points))
					self.last_board = board
		except:
			pass
	def __Piece_value(self,name):
		Pieces = {
			'W_P': 1,
			'W_Q': 9,
			'W_K': 0,
			'W_N': 3,
			'W_B': 3,
			'W_R': 5,
			'B_P': -1,
			'B_Q': -9,
			'B_K': -0,
			'B_N': -3,
			'B_B': -3,
			'B_R': -5,
			' ': 0
		}
		return Pieces[name]
	def game_evaluation_plot(self):
		print("Game History "+str(self.history))
		textfile = open("Last_Game_History.txt", "w")
		textfile.write("index__Move__Evaluation"+"\n")
		for index,(element,eval) in enumerate(zip(self.history,self.__game_evaluation_progres)):
			textfile.write(str(index)+"____"+element +"____"+str(eval)+ "\n")
		textfile.close()
		x = np.linspace(0, len(self.__game_evaluation_progres), num=len(self.__game_evaluation_progres))
		plt.xticks(np.arange(0, len(self.__game_evaluation_progres)+10, 1))
		plt.plot(x, self.__game_evaluation_progres, label="Chess Evaluation during recorded Game")
		plt.grid(True)
		plt.ylabel(" Score ")
		plt.xlabel(" Move ")
		plt.legend(loc="lower right")
		plt.text(-1,- 50, str(self.history), horizontalalignment='center', verticalalignment='center')
		plt.title("Chess Evaluation during recorded Game")
		plt.show()