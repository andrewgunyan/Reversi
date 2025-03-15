AI that plays Reversi!

AI that uses minimax and alpha-beta pruning as well as a heuristic evaluation function to play the game reversi (or othello).

Steps to play:
1. Start the server: java Reversi 10
Note that the parameter 10 specifies the number of minutes that each player has of move
throughout the game.
2. Start player 1 (the Human player in this case): java Human localhost 1
See the files for descriptions of the parameters
3. Start player 2 (the Engine): python reversi_python_client.py localhost 2
