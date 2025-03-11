import numpy as np
import random as rand
import reversi

class ReversiBot:
    def __init__(self, move_num):
        self.move_num = move_num

    def heuristic_evaluation(self, state): 
        my_color = 1                        # assume I am player one. Fix that later once I understand more
        opp_color = 2
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

        print(self.heuristic_evaluation(state))

        valid_moves = state.get_valid_moves()
        move = rand.choice(valid_moves)

        return move

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