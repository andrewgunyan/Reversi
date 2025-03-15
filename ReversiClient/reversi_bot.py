import numpy as np
import random as rand
import json
import reversi
from WThorParser import WThorParser

class ReversiBot:
    def __init__(self, move_num):
        self.move_num = move_num
        self.my_color = move_num
        self.opp_color = 3 - move_num
        self.opening_book = self.load_opening_book("wthor_opening_book.json")

        if not self.opening_book:
                print("Generating WThor opening book...")
                parser = WThorParser("WTH_2021.wtb")
                parser.parse_wthor()
                parser.save_opening_book()
                self.opening_book = self.load_opening_book("wthor_opening_book.json")

    def load_opening_book(self, filename="wthor_opening_book.json"):
        """Loads the opening book from a JSON file."""
        try:
            with open(filename, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def get_move_sequence(self, state):
        """Generate a move sequence string from the game history, matching WThor format."""
        move_sequence = ""
        move_count = 0

        # Iterate over the board and count the number of moves made so far
        for row in range(8):
            for col in range(8):
                if state.board[row][col] != 0:  # If there's a piece
                    move_count += 1

        # Generate the move sequence based on the number of moves made
        for i in range(move_count):
            move_sequence += str(i)  # Append the move count as a string

        print("Current Move Sequence:", move_sequence)
        return move_sequence


    def get_best_move(self, state):
        """Uses WThor opening book before falling back to Minimax."""
        move_sequence = self.get_move_sequence(state)  # Get move sequence

        if move_sequence in self.opening_book:
            print("Using WThor opening book move.")
            move_str = max(self.opening_book[move_sequence], key=self.opening_book[move_sequence].get)
            
            # Convert stored move string back into a tuple
            move = eval(move_str)  # Converts "'(3, 3)'" → (3, 3)
            return move

        print("Minimax (no opening book move found).")
        return self.alpha_beta_search(state)  # Default to Minimax if no book move exists


    #-------------------------------------Alpha-Beta-Pruning--------------------------------------------------#
    def cutoff_test(self, state, depth, max_depth = 5):
        return depth >= max_depth or len(state.get_valid_moves()) == 0

    def alpha_beta_search(self, state):
        best_action = None
        v = -float('inf')
        alpha, beta = -float('inf'), float('inf')

        for action, successor in state.successors():
            min_val = self.min_value(successor, alpha, beta, depth=1)
            if min_val > v:
                v = min_val
                best_action = action
        return best_action
    
    def max_value(self, state, alpha, beta, depth):
        if self.cutoff_test(state, depth):
            return self.heuristic_evaluation(state)
        v = -float('inf')
        for action, successor in state.successors():
            v = max(v, self.min_value(successor, alpha, beta, depth+1))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v
    
    def min_value(self, state, alpha, beta, depth):
        if self.cutoff_test(state, depth):
            return self.heuristic_evaluation(state)
        v = float('inf')
        for action, successor in state.successors():
            v = min(v, self.max_value(successor, alpha, beta, depth+1))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    def heuristic_evaluation(self, state): 
        my_color = self.my_color
        opp_color = self.opp_color
        p = c = l = m = f = d = 0
        my_tiles = opp_tiles = my_frontier = opp_frontier = x = y = 0

        x1 = [-1, -1, 0, 1, 1, 1, 0, -1]  # possible directions for exploring the board
        y1 = [0, 1, 1, 1, 0, -1, -1, -1]  # up, down, left, right, diagnols

        V = [                               # Evaluation of each board spot
            [20, -3, 11, 8, 8, 11, -3, 20],
            [-3, -7, -4, 1, 1, -4, -7, -3],
            [11, -4, 2, 2, 2, 2, -4, 11],
            [8, 1, 2, -3, -3, 2, 1, 8],
            [8, 1, 2, -3, -3, 2, 1, 8],
            [11, -4, 2, 2, 2, 2, -4, 11],
            [-3, -7, -4, 1, 1, -4, -7, -3],
            [20, -3, 11, 8, 8, 11, -3, 20]
        ]

        board = state.board
        for i in range(0, 8):
            for j in range(0, 8):
                if board[i][j] == my_color:
                    d += V[i][j]
                    my_tiles += 1
                elif board[i][j] == opp_color:
                    d -= V[i][j]
                    opp_tiles += 1
                if board[i][j] != 0:        # Calculate the frontier scores
                    for k in range(0, 8):
                        x = i + x1[k]
                        y = j + y1[k]
                        if x >= 0 and x < 8 and y >= 0 and y < 8 and board[x][y] == 0:
                            if board[i][j] == my_color:
                                my_frontier += 1
                            else:
                                opp_frontier += 1
                            break
        if my_tiles > opp_tiles:        # Calculate the piece score difference
            p = (100.0 * my_tiles) / (my_tiles + opp_tiles)
        elif my_tiles < opp_tiles:
            p = -(100.0 * opp_tiles) / (my_tiles + opp_tiles)
        else:
            p = 0

        if my_frontier > opp_frontier:  # Calculate the frontier score difference
            f = -(100.0 * opp_frontier) / (my_frontier + opp_frontier)
        elif my_frontier < opp_frontier:
            f = (100.0 * opp_frontier) / (my_frontier + opp_frontier)
        else:
            f = 0

        my_tiles = opp_tiles = 0    # Corner occupancy evaluation
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        for x, y in corners:
            if board[x][y] == my_color:
                my_tiles += 1
            elif board[x][y] == opp_color:
                opp_tiles += 1
        c = 25 * (my_tiles - opp_tiles)

        my_tiles = opp_tiles = 0    # Corner closeness evaluation
        if board[0][0] == 0:     # Top left corner
            if board[0][1] == my_color:
                my_tiles += 1
            elif board[0][1] == opp_color:
                opp_tiles += 1
            if board[1][1] == my_color:
                my_tiles += 1
            elif board[1][1] == opp_color:
                opp_tiles += 1
            if board[1][0] == my_color:
                my_tiles += 1
            elif board[1][0] == opp_color:
                opp_tiles += 1

        if board[0][7] == 0:    # Top right corner
            if board[0][6] == my_color:
                my_tiles += 1
            elif board[0][6] == opp_color:
                opp_tiles += 1
            if board[1][6] == my_color:
                my_tiles += 1
            elif board[1][6] == opp_color:
                opp_tiles += 1
            if board[1][7] == my_color:
                my_tiles += 1
            elif board[1][7] == opp_color:
                opp_tiles += 1

        if board[7][7] == 0:    # Bottom-right corner
            if board[7][6] == my_color:
                my_tiles += 1
            elif board[7][6] == opp_color:
                opp_tiles += 1
            if board[6][6] == my_color:
                my_tiles += 1
            elif board[6][6] == opp_color:
                opp_tiles += 1
            if board[6][7] == my_color:
                my_tiles += 1
            elif board[6][7] == opp_color:
                opp_tiles += 1
        
        if board[7][0] == 0:    # Bottom-left corner
            if board[7][1] == my_color:
                my_tiles += 1
            elif board[7][1] == opp_color:
                opp_tiles += 1
            if board[6][1] == my_color:
                my_tiles += 1
            elif board[6][1] == opp_color:
                opp_tiles += 1
            if board[6][0] == my_color:
                my_tiles += 1
            elif board[6][0] == opp_color:
                opp_tiles += 1
        l = -12.5 * (my_tiles - opp_tiles)

        # Mobility evaluation
        my_valid_moves = len(state.get_valid_moves())   # Find number of valid moves for me
        original_turn = state.turn
        state.turn = 3 - original_turn
        opponent_valid_moves = len(state.get_valid_moves()) # Find number of valid moves for opponent
        state.turn = original_turn

        if my_valid_moves > opponent_valid_moves:
            m = (100.0*my_valid_moves)/(my_valid_moves+opponent_valid_moves)
        elif my_valid_moves < opponent_valid_moves:
            m = -(100.0*opponent_valid_moves)/(my_valid_moves+opponent_valid_moves)
        else:
            m = 0

        score = (10 * p) + (801.724 * c) + (382.026 * l) + (78.922 * m) + (74.396 * f) + (10 * d)
        return score

    def make_move(self, state):
        try:
            move = self.get_best_move(state)
            if move is not None:
                state.make_move(move)  # Apply move to update the board
                return move
        except Exception as e:
            print(f"Error in get_best_move: {e}")
        
        print("Not a book move, using Minimax.")
        move = self.alpha_beta_search(state)
        if move is not None:
            state.make_move(move)  # Apply move to update the board
        return move


'''
This is the only function that needs to be implemented for the lab!
The bot should take a game state and return a move.

The parameter "state" is of type ReversiGameState and has two useful
member variables. The first is "board", which is an 8x8 numpy array
of 0s, 1s, and 2s. If a spot has a 0 that means it is unoccupied. If
there is a 1 that means the spot has one of player 1's stones. If
there is a 2 on the spot that means that spot has one of player 2's
stones. The other useful member variable is "turn", which is 1 if it's
player 1's turn and 2 if it's player 2's turn.

ReversiGameState objects have a nice method called get_valid_moves.
When you invoke it on a ReversiGameState object a list of valid
moves for that state is returned in the form of a list of tuples.

Move should be a tuple (row, col) of the move you want the bot to make.
'''

'''
Pseudocode
def minimax(state, depth, maximizing_player):
    if depth == 0 or game_over(state):
        return heuristic_evaluation(state)

    if maximizing_player:
        max_eval = -float('inf')
        for move in get_valid_moves(state):
            eval = minimax(apply_move(state, move), depth - 1, False)
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float('inf')
        for move in get_valid_moves(state):
            eval = minimax(apply_move(state, move), depth - 1, True)
            min_eval = min(min_eval, eval)
        return min_eval
'''