import random

# from typing import Deque
import chess

from helpers.chessBoard import posFromFen

# import os


class ChessGame:
    def __init__(self):
        self.fens = [
            "r6r/6R1/p6p/7k/5R2/6PP/2P5/2K5 w - - 0 1",
            "r6r/p5R1/7p/7k/5R2/7P/2P3P1/2K5 w - - 0 1",
            "r6r/p1p3R1/3n1R1p/8/7k/7P/2P3P1/2K5 w - - 0 1",
            "5rk1/4Pp1p/4q1p1/2p5/6P1/3r3P/6B1/Q4RK1 w - - 0 1",
            "5r1k/4Pp1p/4q1p1/2p5/6P1/3r3P/6B1/5RK1 w - - 0 1",
            "8/8/8/8/8/4k3/8/6KQ w - - 0 1",
            "q7/8/8/8/8/4k3/8/6K1 w - - 0 1",
            "3k4/qp4b1/4K3/8/8/5b2/8/8 w - - 0 1",
            "3k4/1p3Kb1/8/8/8/5b2/8/6q1 w - - 0 1",
            "3k2K1/1p4b1/8/8/4b3/8/8/6q1 w - - 0 1",
            "8/2n2B2/p4p2/1p3k1p/P4P2/1P2K1PP/8/8 w - - 0 1",
            "1k6/8/8/8/8/5K2/6R1/7R w - - 0 1",
            "1k6/8/8/8/8/5K2/R7/7R w - - 0 1",
            "k7/2K5/7R/8/8/8/8/8 w - - 0 1",
            "1k6/8/7R/2K5/8/8/8/8 w - - 0 1",
            "8/8/8/8/r7/4k3/8/7K w - - 0 1",
            "8/8/8/8/4k3/8/4q3/7K w - - 0 1",
            "8/8/8/8/4k3/8/1q6/7K w - - 0 1",
            "8/8/8/1q6/4k3/8/8/7K w - - 0 1",
            "k7/4P3/2K5/8/8/8/8/8 w - - 0 1",
            "8/8/5R2/8/2kNKP2/P7/5P2/8 w - - 0 1",
            "8/8/5R2/8/2KNkP2/P7/5P2/8 w - - 0 1",
            "r1b2rk1/2p2ppp/p7/1p6/3P3q/1BP3bP/PP3QP1/RNB1R1K1 w - - 0 1",
        ]

        self.rowCount = 8
        self.colCount = 8
        ChessGame.actionSize = (self.rowCount * self.colCount) ** 2
        self.board = chess.Board()
        # self.memory = Deque(maxlen=524288)

    def getInitialState(self):
        self.board.set_fen(self.fens[random.randint(0, len(self.fens) - 1)])
        # self.board = board
        state = posFromFen(self.board.fen())
        return (state, self.board)  # TODO

    def checkWin(self):
        pass
