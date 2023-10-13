import sys
from BetaZeroParallel import BetaZeroParallel
from ChessGame import ChessGame

from MCTS import MCTS
import numpy as np
from chess import Move
import torch
from BetaZero import BetaZero
from NN import ResNet

import json

np.set_printoptions(threshold=sys.maxsize) 

with open("config.json", "r") as f:
    args = json.load(f)


chessGame = ChessGame(args["numParallelGames"])
model = ResNet(chessGame, 16, 64, args["device"])
if "model" in args and args["model"] != "":
    model.load_state_dict(torch.load(args["model"]))
# model.eval()

player = True 

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
if "optimizer" in args and args["optimizer"] != "":
    optimizer.load_state_dict(torch.load(args["optimizer"]))

if args["numParallelGames"] > 1:
    betaZero = BetaZeroParallel(model, optimizer, chessGame, args)
else:
    betaZero = BetaZero(model, optimizer, chessGame, args)

betaZero.learn()

mcts = MCTS( model, chessGame, args)
state, board = chessGame.getInitialState()
tensorState = torch.tensor(chessGame.getEncodedState(state)).unsqueeze(0)

while True:
    print(board)
    if player == False:
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
    state, board = chessGame.getNextState(state, action, board, player)
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
    
    