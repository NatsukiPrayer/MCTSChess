from classes.BetaZeroParallel import BetaZeroParallel
from classes.ChessGame import ChessGame
from classes.MCTS import MCTS
from classes.BetaZero import BetaZero
from classes.NN import ResNet

import helpers.argsParser

from chess import Move
import numpy as np
import torch

import json
import sys


def train(model: ResNet, game: ChessGame, config: dict):
    if "model" in config and config["model"] != "":
        model.load_state_dict(torch.load(config["model"], map_location=config["device"]))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0001)
    if "optimizer" in config and config["optimizer"] != "":
        optimizer.load_state_dict(torch.load(config["optimizer"]))

    # TODO: убрать этот if объеденив классы в 1
    if config["numParallelGames"] > 1:
        betaZero = BetaZeroParallel(model, optimizer, game, config)
    else:
        betaZero = BetaZero(model, optimizer, game, config)

    betaZero.learn()


def test(model: ResNet, game: ChessGame, config: dict):
    model.eval()
    mcts = MCTS(model, game, config)
    state, board = game.getInitialState()

    player = True
    while True:
        print(board)
        if not player:
            validMoves = game.getValidMoves(board)
            print("valid moves", validMoves)
            action = input()
            if not Move.from_uci(action) in validMoves:
                print("Invalid move")
                continue
        else:
            neutralState = game.changePerspective(state)
            mctsProbs = mcts.search(neutralState, board, idx=0)
            action = np.argmax(mctsProbs)
        state, board = game.getNextState(state, action, board, player)
        value, isTerminal = game.getValAndTerminate(board)
        if isTerminal:
            print(board)
            if value == 1:
                print(player, "won")
            else:
                print("draw")
            break
        player = not player
        print()


def main(*args, **kwargs):
    args = helpers.argsParser.parse(*args, **kwargs)
    np.set_printoptions(threshold=sys.maxsize)

    with open(args.config, "r") as f:
        config = json.load(f)

    chessGame = ChessGame(config["numParallelGames"])
    # TODO: Create config for model
    model = ResNet(chessGame, 16, 64, config["device"])
    if args.mode == "train":
        train(model, chessGame, config)
    else:
        test(model, chessGame, config)


if __name__ == "__main__":
    main()
