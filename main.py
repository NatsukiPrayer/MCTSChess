from ChessGame import ChessGame
from MCTS import MCTS
import numpy as np
from chess import Move

args = {'C':1.41, 'num_searches':1000}

chessGame = ChessGame()
mcts = MCTS(chessGame, args)
state, board = chessGame.getInitialState()
player = True
while True:
    print(board)
    if player == True:
        validMoves = chessGame.getValidMoves(board)
        print("valid moves", validMoves)
        action = input()
        if not Move.from_uci(action) in validMoves:
            print("Invalid move")
            continue
    
    else:
        neutralState = chessGame.changePerspective(state)
        mctsProbs = mcts.search(neutralState, board)
        action = np.argmax(mctsProbs)
    state, board = chessGame.getNextState(state, action, board)
    value, isTerminal = chessGame.getValAndTerminate(board)
    if isTerminal:
        print(board)
        if value == 1:
            print(player, 'won')
        else:
            print('draw')
        break
    player = not player
    
    