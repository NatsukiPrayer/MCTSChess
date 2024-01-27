from classes.ChessGame import ChessGame

# from classes.MCTS import MCTS
from classes.BetaZero import BetaZero
from classes.NN import ResNet

import helpers.argsParser

# from chess import Move
import numpy as np
import torch

import json
import sys

# from helpers.chessBoard import changePerspective, getNextState, getValAndTerminate, getValidMoves


def train(model: ResNet, game: ChessGame, config: dict):
    if "model" in config and config["model"] != "":
        model.load_state_dict(torch.load(config["model"], map_location=config["device"]))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0001)
    if "optimizer" in config and config["optimizer"] != "":
        optimizer.load_state_dict(torch.load(config["optimizer"]))

    betaZero = BetaZero(model, optimizer, config)

    betaZero.learn()


# def test(model: ResNet, game: ChessGame, config: dict):
#     model.eval()
#     mcts = MCTS(model, config)
#     state, board = game.getInitialState()

#     player = True
#     while True:
#         print(board)
#         if not player:
#             validMoves = getValidMoves(board)
#             print("valid moves", validMoves)
#             action = input()
#             if not Move.from_uci(action) in validMoves:
#                 print("Invalid move")
#                 continue
#         else:
#             neutralState = changePerspective(state)
#             mctsProbs = mcts.search(
#                 neutralState,
#             )
#             action = np.argmax(mctsProbs)
#         state, board = getNextState(state, action, board, player)
#         value, isTerminal = getValAndTerminate(board)
#         if isTerminal:
#             print(board)
#             if value == 1:
#                 print(player, "won")
#             else:
#                 print("draw")
#             break
#         player = not player
#         print()


def main(*args, **kwargs):
    args = helpers.argsParser.parse(*args, **kwargs)
    np.set_printoptions(threshold=sys.maxsize)

    with open(args.config, "r") as f:
        config = json.load(f)

    chessGame = ChessGame()
    # TODO: Create config for model
    model = ResNet(chessGame, 32, 128, config["device"])
    if args.mode == "train":
        train(model, chessGame, config)
    # else:
    #     test(model, chessGame, config)


if __name__ == "__main__":
    main()
