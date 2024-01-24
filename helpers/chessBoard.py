import chess
import numpy as np
from numpy.typing import NDArray

letters = ("a", "b", "c", "d", "e", "f", "g", "h")
figures = ("", "p", "r", "n", "b", "q", "k")


def encode(inp: str):
    action = str(inp)
    rowFrom = int(action[1]) - 1
    colFrom = letters.index(action[0]) * 8
    rowWhere = (int(action[3]) - 1) * 64
    colWhere = letters.index(action[2]) * 512
    return rowFrom + rowWhere + colFrom + colWhere


def decode(action: int, color: bool):
    if not color:
        action = 4095 - action
    rowFrom = action % 8
    colFrom = action // 8 % 8
    rowWhere = (action // 64) % 8
    colWhere = (action // 512) % 8
    return rowFrom, colFrom, rowWhere, colWhere


def getNextState(state: NDArray, action: int | str, board: chess.Board, color: bool):  # TODO: types; fix this
    try:
        rowFrom, colFrom, rowWhere, colWhere = decode(int(action), color)
        action = f"{letters[colFrom]}{rowFrom+1}{letters[colWhere]}{rowWhere+1}"
    except Exception:
        action = str(action)
        rowFrom = int(action[1]) - 1
        colFrom = letters.index(action[0])
        rowWhere = int(action[3]) - 1
        colWhere = letters.index(action[2])

    try:
        board.push_uci(action)
    except chess.IllegalMoveError:
        try:
            action = action + "q"
            board.push_uci(action)
        except Exception:
            print("Chuyali???Vonyaet")

    state = posFromFen(board.fen())
    if not color:
        state = changePerspective(state)
    return (state, board)


def getValAndTerminate(board: chess.Board):
    if board.is_checkmate():
        return (1, True)
    elif (
        board.is_variant_end()
        or board.is_stalemate()
        or board.is_repetition()
        or board.is_variant_draw()
        or board.is_insufficient_material()
        or board.is_fifty_moves()
    ):
        return (0, True)
    return (0, False)


def getValidMoves(board):
    return list(board.legal_moves)


def changePerspective(state):
    if len(state.shape) == 3:
        return np.rot90((state * -1), 2, axes=(1, 2))
    return np.rot90((state * -1), 2)


def posFromFen(fen: str):
    fen = fen.split(" ", maxsplit=1)[0].split("/")  # type: ignore
    pos = np.zeros((8, 8))
    for idx, row in enumerate(fen):
        j = 0
        i = 0
        while i < len(row):
            if row[i].isdigit():
                j += int(row[i])
            else:
                pos[idx][j] = figures.index(row[i].lower()) * (-1 if row[i].islower() else 1)
                j += 1
            i += 1

    return pos


def getEncodedState(self, state):
    encodedState = np.stack(
        (
            state == -6,
            state == -5,
            state == -4,
            state == -3,
            state == -2,
            state == -1,
            state == 0,
            state == 6,
            state == 5,
            state == 4,
            state == 3,
            state == 2,
            state == 1,
        )
    ).astype(np.float32)
    if len(state.shape) == 3:
        encodedState = np.swapaxes(encodedState, 0, 1)
    return encodedState
