from classes.ChessGame import ChessGame
import numpy as np
import chess
import torch
from tqdm import tqdm

from classes.Node import Node
from classes.Drawer import Drawer
from helpers.encode import encode


class MCTS:
    def __init__(self, model, game: ChessGame, args):
        self.game = game
        self.args = args
        self.model = model
        self.drawer = None

    @torch.no_grad()
    def search(self, state, board, idx):
        root = Node(self.game, self.args, state, board, prior=1)
        self.drawer = Drawer(root)

        for iter in (num_searches := tqdm(range(self.args["num_searches"]), leave=False)):
            num_searches.set_description(f"Searches {idx}")
            node = root

            # while node.isFullyExpanded():
            #     node = node.select()

            while len(node.children) > 0:
                node = node.select()

            # self.drawer.update(root)

            value, isTerminal = self.game.getValAndTerminate(node.board)
            value = value * -1
            if not isTerminal:
                policy, value = self.model(
                    torch.tensor(self.game.getEncodedState(node.state), device=self.args["device"]).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()  # type: ignore
                validMoves = self.game.getValidMoves(node.board)
                zeros = np.zeros(4096)
                for move in validMoves:
                    zeros[encode(str(move))] = 1
                if node.board.turn == chess.BLACK:
                    zeros = np.flip(zeros)
                policy *= zeros
                policy /= np.sum(policy)

                value = value.item()

                node = node.expand(policy, iter, node.board.turn == chess.WHITE)

            node.backpropogate(value, iter)

        # self.drawer.update(root)
        actionProbs = np.zeros(self.game.actionSize)
        if len(root.children) == 0:
            raise Exception("GameEnd???")
        for child in root.children:
            actionProbs[child.actionTaken] = child.visitCount
        a = np.sum(actionProbs)
        actionProbs /= a
        return actionProbs

    # return visit counts
