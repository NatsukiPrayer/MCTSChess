import sys
from classes.BetaZeroParallel import BetaZeroParallel
from classes.ChessGame import ChessGame

from classes.MCTS import MCTS
import numpy as np
from chess import Move
import torch
from classes.BetaZero import BetaZero
from classes.NN import ResNet

import json


if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)

    with open("configs/config.json", "r") as f:
        args = json.load(f)

    chessGame = ChessGame(args["numParallelGames"])

    # model = ResNet(chessGame, 32, 128, args["device"])
    model = ResNet(chessGame, 16, 64, args["device"])

    if "model" in args and args["model"] != "":
        model.load_state_dict(torch.load(args["model"], map_location=args["device"]))
    # model.eval()

    player = True

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0001)
    if "optimizer" in args and args["optimizer"] != "":
        optimizer.load_state_dict(torch.load(args["optimizer"]))

    if args["numParallelGames"] > 0:
        betaZero = BetaZeroParallel(model, optimizer, chessGame, args)
    else:
        betaZero = BetaZero(model, optimizer, chessGame, args)

    betaZero.learn()

    mcts = MCTS(model, chessGame, args)
    state, board = chessGame.getInitialState()
    tensorState = torch.tensor(chessGame.getEncodedState(state)).unsqueeze(0)

    while True:
        print(board)
        if not player:
            validMoves = chessGame.getValidMoves(board)
            print("valid moves", validMoves)
            action = input()
            if not Move.from_uci(action) in validMoves:
                continue
                print("Invalid move")

        else:
            neutralState = chessGame.changePerspective(state)
            mctsProbs = mcts.search(neutralState, board, idx=0)
            action = np.argmax(mctsProbs)
        state, board = chessGame.getNextState(state, action, board, player)
        value, isTerminal = chessGame.getValAndTerminate(board)
        if isTerminal:
            print(board)
            if value == 1:
                print(player, "won")
            else:
                print("draw")
            break
        player = not player
        print()
