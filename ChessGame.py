import chess 
import numpy as np

class ChessGame:
    def __init__(self, numParallel):
        self.rowCount = 8
        self.colCount = 8
        self.actionSize = (self.rowCount * self.colCount)**2
        self.board = chess.Board()
        self.numParallel = numParallel
        
        self.letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

    def getInitialState(self):
        wPawns = [1 for i in range(self.colCount)]
        bPawns = [-1 for i in range(self.colCount)]
        mainWPieces = [2,3,4,5,6,4,3,2]
        mainBPieces = [-2,-3,-4,-5,-6,-4,-3,-2]
        if self.numParallel > 1:
            board = chess.Board()
            
        else:
            self.board.reset()
            board = self.board
        self.board.set_fen('8/8/8/4k3/R7/8/4K3/4R3 b - - 0 1')
        zeros = [0 for i in range(self.colCount)]
        state = np.array([mainWPieces,
                         wPawns,
                         zeros,
                         zeros,
                         zeros,
                         zeros,
                         bPawns,
                         mainBPieces])
        state = np.array([zeros, 
                          zeros, 
                          zeros,
                          [0, 0, 0, 0, -6, 0, 0, 0],
                          [2,0,0,0,0,0,0,0],
                          zeros,
                          [0,0,0,0,6,0,0,0],
                          [0,0,0,0,2,0,0,0]])
        return (state, board)
    
    def getNextState(self, state, action, board):
        try:
            action = int(action)
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
    
        fig = state[rowFrom, colFrom]
        uciMove = f'{self.letters[colFrom]}{rowFrom+1}{self.letters[colWhere]}{rowWhere+1}' 
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
        if board.is_checkmate():
            return (1, True)
        elif board.is_variant_end() or board.is_stalemate() or board.is_repetition() or board.is_variant_draw() or board.is_insufficient_material() or board.is_fifty_moves():
            return (0, True)
        return (0, False)
         
    def getValidMoves(self, board):
        return list(board.legal_moves)

    def changePerspective(self, state):
        return np.rot90(np.rot90((state * -1)))
    
    def getEncodedState(self, state):
        encodedState = np.stack((state == -6,state == -5,state == -4,state == -3,state == -2,state == -1,state == 0,state == 6,state == 5,state == 4,state == 3,state == 2,state == 1)).astype(np.float32)
        if len(state.shape) == 3:
            encodedState = np.swapaxes(encodedState, 0, 1)
        return encodedState 



