import random
import chess 
import numpy as np

figures = ('', 'p', 'r', 'n', 'b', 'q', 'k')

class ChessGame:
    def __init__(self, numParallel):
        self.rowCount = 8
        self.fens = [
            "r6r/p7/5R1p/7k/6R1/7P/2P3P1/2K5 w - - 0 1",
            "r6r/p5R1/5R1p/8/7k/7P/2P3P1/2K5 w - - 0 1",
            "r6r/p1p3R1/3n1R1p/8/7k/7P/2P3P1/2K5 w - - 0 1",
            "5rk1/4Pp1p/4q1p1/2p5/6P1/3r3P/6B1/Q4RK1 w - - 0 1",
            "5r1k/4Pp1p/4q1p1/2p5/6P1/3r3P/6B1/5RK1 w - - 0 1",
            "8/8/8/8/8/4k3/8/6KQ w - - 0 1",
            "q7/8/8/8/8/4k3/8/6K1 w - - 0 1",
            "3k4/qp4b1/4K3/8/8/5b2/8/8 w - - 0 1",
            "3k4/1p3Kb1/8/8/8/5b2/8/6q1 w - - 0 1",
            "3k2K1/1p4b1/8/8/4b3/8/8/6q1 w - - 0 1",
            "8/2n2B2/p4p2/1p3k1p/P4P2/1P2K1PP/8/8 w - - 0 1"
            ]
        self.colCount = 8
        self.actionSize = (self.rowCount * self.colCount)**2
        self.board = chess.Board()
        self.numParallel = numParallel
        
        self.letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

    def posFromFen(self, fen: str):
        fen = fen.split(' ', maxsplit=1)[0].split('/')
        pos = np.zeros((8,8))
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

    def getInitialState(self):
        
        if self.numParallel > 1:
            board = chess.Board()
            
        else:
            self.board.reset()
            board = self.board
        board.set_fen(self.fens[random.randint(0, len(self.fens)-1)])
        # self.board = board
        state = self.posFromFen(board.fen())
        return (state, board)
    
    def decode(self, action, color):
        if not color:
            action = 4095 - action
        rowFrom = action % 8
        colFrom = action // 8 % 8
        rowWhere = (action // 64) % 8
        colWhere = (action // 512) % 8 
        return rowFrom, colFrom, rowWhere, colWhere

    def getNextState(self, state, action, board, color): #TODO: fix this
        
        try:
            rowFrom, colFrom, rowWhere, colWhere = self.decode(int(action), color)
            uciRowFrom, uciColFrom, uciRowWhere, uciColWhere = self.decode(int(action), color)
            if not color:
                action = 4095 - action
            rowFrom = action % 8
            colFrom = action // 8 % 8
            rowWhere = (action // 64) % 8
            colWhere = (action // 512) % 8 
        except:
            action = str(action)
            rowFrom = int(action[1]) - 1
            colFrom = self.letters.index(action[0])
            rowWhere = int(action[3]) - 1
            colWhere = self.letters.index(action[2])  

        uciMove = f'{self.letters[uciColFrom]}{uciRowFrom+1}{self.letters[uciColWhere]}{uciRowWhere+1}' 
        if color:
            rowFrom = 7 - rowFrom
            rowWhere = 7 - rowWhere
        else:
            colFrom = 7 - colFrom
            colWhere = 7 - colWhere
            
        fig = state[rowFrom, colFrom]
        assert fig > 0, "Figure not choosen"
        
        state[rowFrom, colFrom] = 0
        try:
            board.push_uci(uciMove)
            state[rowWhere, colWhere] = fig
        except chess.IllegalMoveError:
            try:
                uciMove = uciMove + 'q'
                board.push_uci(uciMove)
                state[rowWhere, colWhere] = 5
            except:
                print('Chuyali???Vonyaet')
        return (state, board)
    
    def checkWin(self):
            pass
    
    def getValAndTerminate(self, board: chess.Board):
        if board.is_checkmate():
            return (1, True)
        elif board.is_variant_end() or board.is_stalemate() or board.is_repetition() or board.is_variant_draw() or board.is_insufficient_material() or board.is_fifty_moves():
            return (0, True)
        return (0, False)
         
    def getValidMoves(self, board):
        return list(board.legal_moves)

    def changePerspective(self, state):
        if len(state.shape) == 3: 
            return np.rot90((state * -1), 2, axes=(1,2))
        return np.rot90((state * -1), 2)
    
    def getEncodedState(self, state):
        encodedState = np.stack((state == -6,state == -5,state == -4,state == -3,state == -2,state == -1,state == 0,state == 6,state == 5,state == 4,state == 3,state == 2,state == 1)).astype(np.float32)
        if len(state.shape) == 3:
            encodedState = np.swapaxes(encodedState, 0, 1)
        return encodedState 



