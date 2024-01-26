from classes.ChessGame import ChessGame
from classes.Node import Node


class SPG:
    def __init__(self, game: ChessGame) -> None:
        self.state, self.board = game.getInitialState()
        self.memory = []  # ?! TDDO: А какой тут тип лежит?
        self.root: Node = Node(self.state, self.board)
        self.node: Node | None = None
