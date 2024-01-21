from classes.ChessGame import ChessGame


class SPG:
    def __init__(self, game: ChessGame) -> None:
        self.state, self.board = game.getInitialState()
        self.memory = []
        self.root = None
        self.node = None
