import chess 
import numpy as np

class ChessGame:
    def __init__(self):
        self.rowCount = 8
        self.colCount = 8
        self.actionSize = (self.rowCount * self.colCount)**2
        self.board = chess.Board()
        self.letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

    def getInitialState(self):
        wPawns = [1 for i in range(self.colCount)]
        bPawns = [-1 for i in range(self.colCount)]
        mainWPieces = [2,3,4,5,6,4,3,2]
        mainBPieces = [-2,-3,-4,-5,-6,-4,-3,-2]
        return (np.array([mainWPieces,
                         wPawns,
                         [0 for i in range(self.colCount)],
                         [0 for i in range(self.colCount)],
                         [0 for i in range(self.colCount)],
                         [0 for i in range(self.colCount)],
                         [0 for i in range(self.colCount)],
                         bPawns,
                         mainBPieces]), self.board)
    
    def getNextState(self, state, action, board):
        try:
            action = int(action)
            rowFrom = (action // 64) // 8
            colFrom = (action // 64) % 8
            rowWhere = (action % 64) // 8
            colWhere = (action % 64) % 8 
        except:
            action = str(action)
            rowFrom = int(action[1]) - 1
            colFrom = self.letters.index(action[0])
            rowWhere = int(action[3]) - 1
            colWhere = self.letters.index(action[2])  
    
        fig = state[rowFrom, colFrom]
        uciMove = f'{self.letters[colFrom]}{rowFrom+1}{self.letters[colWhere]}{rowWhere+1}' 
        print(list(board.legal_moves))
        print(board.is_variant_end() or board.is_stalemate() or board.is_repetition() or board.is_variant_draw() or board.is_insufficient_material() or board.is_fifty_moves())
        print(board)
        try:
            board.push_uci(uciMove)
        except chess.IllegalMoveError:
            try:
                uciMove = uciMove + 'q'
                board.push_uci(uciMove)
            except:
                print('Chuyali???Vonyaet')
        state[rowFrom, colFrom] = 0
        state[rowWhere, colWhere] = fig
        return (state, board)
    
    def checkWin(self):
            pass
    
    def getValAndTerminate(self, board: chess.Board):
        if board.is_variant_end() or board.is_stalemate() or board.is_repetition() or board.is_variant_draw() or board.is_insufficient_material() or board.is_fifty_moves():
            if (
            board.is_stalemate() or board.is_repetition() or board.is_variant_draw() or board.is_insufficient_material() or board.is_fifty_moves()
            ):
                return (0, True)
            elif board.is_checkmate():
                return (1, True)
        return (0, False)
         

    def getValidMoves(self, board):
        return list(board.legal_moves)

    def changePerspective(self, state):
        return np.rot90(np.rot90((state * -1)))



