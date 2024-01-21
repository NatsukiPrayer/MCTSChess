letters = ["a", "b", "c", "d", "e", "f", "g", "h"]


def encode(inp: str):
    action = str(inp)
    rowFrom = int(action[1]) - 1
    colFrom = letters.index(action[0]) * 8
    rowWhere = (int(action[3]) - 1) * 64
    colWhere = letters.index(action[2]) * 512
    return rowFrom + rowWhere + colFrom + colWhere
