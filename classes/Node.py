import math
import chess
from numpy import floating
from numpy.typing import NDArray
from helpers.chessBoard import changePerspective, getNextState, decode, letters  # , encode


class Node:
    C = 0

    @property
    def val(self):
        try:
            v = self.valueSum.item()  # type: ignore
        except ValueError:
            v = self.valueSum
        return f"{str(self.board)}\nVisits: {self.visitCount}\n" f"Value: {v:.3f}\nPrior: {self.prior:.3f}"

    def __init__(
        self,
        state: NDArray[floating],
        board: chess.Board,
        parent: "Node | None" = None,
        actionTaken: int | str | None = None,
        prior: float = 0,
        visitCount: int = 0,
    ):
        self.state = state
        self.board = board
        self.parent = parent
        self.valueSum = 0
        self.visitCount = visitCount
        self.prior = prior
        self.ucb = self.prior * self.C

        # if isinstance(actionTaken, int) or actionTaken is None:
        #     self.actionTaken = actionTaken
        # else:
        #     self.actionTaken = encode(actionTaken)  # type: ignore

        self.actionTaken = actionTaken

        self.children: list[Node] = []
        self.noChildren: list[Node] = []

    def __repr__(self) -> str:
        return f"UCB = {self.ucb} visits = {self.visitCount}\nval = {self.valueSum}\nprior = {self.prior}"

    def updateUCB(self):
        if self.visitCount == 0:
            qValue = 0
        else:
            qValue = 1 - (self.valueSum / self.visitCount + 1) / 2
        self.ucb = (
            qValue + self.C * (math.sqrt(self.parent.visitCount) / (self.visitCount + 1)) * self.prior  # type: ignore
        )

    def isFullyExpanded(self):
        return len(self.children) > 0

    def select(self):
        return self.children[0]

    def expand(self, spgPolicy: NDArray[floating], color: bool) -> "Node":
        for action, isLegal in enumerate(spgPolicy):
            if isLegal > 0:
                childState = self.state.copy()
                childBoard = self.board.copy()
                childState, childBoard = getNextState(childState, action, childBoard, color)
                childState = changePerspective(childState)
                child = Node(
                    childState,
                    childBoard,
                    self,
                    actionTaken=action,
                    prior=spgPolicy[action],
                )  # TODO: нормально переписать аргументы
                self.noChildren.append(child)

        self.noChildren.sort(key=lambda x: x.ucb, reverse=True)
        self.children = [self.noChildren.pop(0)]

        return self

    def backpropogate(self, value: float):
        self.valueSum += value
        self.visitCount += 1

        value *= -1

        if self.parent is not None:
            self.updateUCB()

            idx = 1
            for child in self.parent.children[1:]:
                child.updateUCB()
                if child.ucb <= self.ucb:
                    break
                idx += 1
                if idx != 1:
                    self.parent.children.insert(idx, self.parent.children.pop(0))

            if len(self.parent.noChildren) > 0:
                self.parent.noChildren[0].updateUCB()
                if self.parent.children[0].ucb < self.parent.noChildren[0].ucb:
                    self.parent.children.insert(0, self.parent.noChildren.pop(0))

            self.parent.backpropogate(value)

    def find(self, action: int) -> "Node":
        # ! TODO: найти доску по action среди детей
        # rowFrom, colFrom, rowWhere, colWhere = decode(action, not self.board.turn)
        # nextBoard = self.board.push_uci(f"{letters[colFrom]}{rowFrom+1}{letters[colWhere]}{rowWhere+1}")
        # for child in self.children:
        #     if child.board == nextBoard:
        #         return child
        # for child in self.noChildren:
        #     if child.board == nextBoard:
        #         return child
        for node in self.children:
            if action == node.actionTaken:
                return node
        for node in self.noChildren:
            if action == node.actionTaken:
                return node
        raise IndexError("No such board")
