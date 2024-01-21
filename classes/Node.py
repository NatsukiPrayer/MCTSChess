import math
import chess
import numpy as np
from classes.ChessGame import ChessGame
from helpers.encode import encode


class Node:
    _ucb = 0

    @property
    def ucb(self):
        return self._ucb

    @ucb.setter
    def ucb(self, value):
        self._ucb = value

    def updateUCB(self, value, idx, search):
        self._ucb = value
        if self.lastUpdate == search:
            return
        self.lastUpdate = search
        if self.parent is not None:
            while idx < len(self.parent.children) - 1:
                value = self.parent.children[idx + 1].getUCB_Self()

                if value > self._ucb:
                    self.parent.children[idx + 1].updateUCB(value, idx + 1, search)
                    self.parent.children[idx], self.parent.children[idx + 1] = (
                        self.parent.children[idx + 1],
                        self.parent.children[idx],
                    )
                    idx += 1
                else:
                    self.parent.children[idx + 1].ucb = value
                    self.parent.children[idx + 1].lastSearch = search

                    break

    def __init__(
        self, game: ChessGame, args, state, board: chess.Board, parent=None, actionTaken=None, prior=0, visitCount=0
    ):
        self.game = game
        self.args = args
        self.state = state
        self.board = board
        self.parent = parent

        if isinstance(actionTaken, int) or actionTaken is None:
            self.actionTaken = actionTaken
        else:
            self.actionTaken = encode(actionTaken)
        self.prior = prior
        self.lastUpdate = -1

        self.children = []

        self.visitCount = visitCount
        self.valueSum = 0

    @property
    def val(self):
        try:
            v = self.valueSum.item()  # type: ignore
        except ValueError:
            v = self.valueSum
        return f"{str(self.board)}\nVisits: {self.visitCount}\nValue: {v:.3f}\nPrior: {self.prior:.3f}"  # TODO

    def isFullyExpanded(self):
        return len(self.children) > 0

    def select(self):
        bestUCB = -np.inf
        bestChild = self.children[0]
        for child in self.children[1:]:
            ucb = self.getUCB(child)
            if ucb > bestUCB:
                bestChild = child
                bestUCB = ucb
        return bestChild

    def getUCB(self, child):
        if child.visitCount == 0:
            qValue = 0
        else:
            qValue = 1 - (child.valueSum / child.visitCount + 1) / 2
            # qValue = -child.valueSum / child.visitCount
        return qValue + self.args["C"] * ((math.sqrt(self.visitCount) / (child.visitCount + 1))) * child.prior

    def getUCB_Self(self):
        if self.visitCount == 0:
            qValue = 0
        else:
            qValue = 1 - (self.valueSum / self.visitCount + 1) / (
                2 * math.sqrt(self.parent.visitCount + 1)  # type: ignore
            )

        # TODO: Почему у нас объявлен parent как None? Можем ли отказаться как-то от этого?

        return qValue + self.args["C"] * ((1 / (self.visitCount + 1))) * self.prior

    def expand(self, policy, mask, color, search=0):
        for action, isLegal in enumerate(mask):
            if isLegal > 0:
                childState = self.state.copy()
                childBoard = self.board.copy()
                childState, childBoard = self.game.getNextState(childState, action, childBoard, color)
                childState = self.game.changePerspective(childState)
                child = Node(self.game, self.args, childState, childBoard, self, action, policy[action])
                self.children.append(child)
                child.updateUCB(child.getUCB_Self(), 0, search)

        return self

    def backpropogate(self, value, search):
        self.valueSum += value
        self.visitCount += 1

        value *= -1
        if self.parent is not None:
            self.updateUCB(self.getUCB_Self(), 0, search)
            self.parent.backpropogate(value, search)
