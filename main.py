import sys
from ChessGame import ChessGame
from MCTS import MCTS
import numpy as np
from chess import Move
import torch
from BetaZero import BetaZero
from NN import ResNet
np.set_printoptions(threshold=sys.maxsize)

args = {'C':2, 'num_searches':1000, 'numIterations':200, 'numSelfPlayIterations':5000, 'numEpochs':32, 'batchSize':128}


chessGame = ChessGame()
model = ResNet(chessGame, 32, 128)
# model.load_state_dict(torch.load('E:\BetaZero\model_99.pt'))
# model.eval()



player = True

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

betaZero = BetaZero(model, optimizer, chessGame, args)
betaZero.learn()

mcts = MCTS( model, chessGame, args)
state, board = chessGame.getInitialState()
tensorState = torch.tensor(chessGame.getEncodedState(state)).unsqueeze(0)

while True:
    print(board)
    if player == True:
        validMoves = chessGame.getValidMoves(board)
        print("valid moves", validMoves)
        action = input()
        if not Move.from_uci(action) in validMoves:
            continue
            print("Invalid move")
    
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
    print()
    
    