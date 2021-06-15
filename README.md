# Chess-vision
 Machine vision script based only on [opencv](https://opencv.org/) library.
<!-- ABOUT THE PROJECT -->
## About The Project

Project includes the implementation of  recognition of chess pieces and chess board  based on the recordings of the games played on the [chess.com](https://www.chess.com/)  website.

Each move of piece is detected. Move is saved to the list,which is keeping the history of the recorded game. Additionally script is evaluating matieral imbalances. After the game is finished, it writes its game history to a file with the .txt extension.
<!-- USAGE EXAMPLES -->
## Recognition of pieces and a chessboard
<p align="center">
<img src="https://user-images.githubusercontent.com/76798626/122093609-866a2f00-ce0b-11eb-8f1d-bb3869d43260.gif">
</p>

## Detection of board squares
<p align="center">
<img width="600" height="500" src="https://user-images.githubusercontent.com/76798626/122094130-1a3bfb00-ce0c-11eb-99af-e7808ec84ce2.png">
</p>

## Evaluation of material imbalance during game
<p align="center">
<img width="600" height="500" src="https://user-images.githubusercontent.com/76798626/122094489-861e6380-ce0c-11eb-9767-f81ca9eb8e61.png">
</p>

## Usage
If  you want to try run this code, you have to put your recorded game in /videos folder.
