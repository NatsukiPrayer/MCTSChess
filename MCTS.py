import math
from ChessGame import ChessGame
import numpy as np
import chess

class Node:
    def __init__(self, game: ChessGame, args, state, board: chess.Board, parent=None, actionTaken = None):
        self.letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        self.game = game
        self.args = args
        self.state = state
        self.board = board
        self.parent = parent
        if isinstance(actionTaken, int) or actionTaken is None:
            self.actionTaken = actionTaken
        else:
            action = str(actionTaken)
            rowFrom = (int(action[1]) - 1) * 64
            colFrom = self.letters.index(action[0]) * 64
            rowWhere = (int(action[3]) - 1) * 64
            colWhere = self.letters.index(action[2]) * 64  
            self.actionTaken = rowFrom + rowWhere + colFrom + colWhere


        self.children = []
        self.expandableMoves = game.getValidMoves(self.board)

        self.visitCount = 0
        self.valueSum = 0

    def isFullyExpanded(self):
        return len(self.expandableMoves) == 0 and len(self.children) > 0

    def select(self):
        bestChild = None
        bestUCB = -np.inf
        for child in self.children:
            ucb = self.getUCB(child)
            if ucb > bestUCB:
                bestChild = child
                bestUCB = ucb
        return bestChild
    
    def getUCB(self, child):
        qValue = 1 - (child.valueSum / child.visitCount + 1) / 2
        return qValue + self.args['C'] * math.sqrt(math.log(self.visitCount) / child.visitCount)


    def expand(self):
        action = np.random.choice(self.expandableMoves)
        self.expandableMoves.pop(self.expandableMoves.index(action))
        childState = self.state.copy()
        childBoard = self.board.copy()
        childState, childBoard  = self.game.getNextState(childState, action, childBoard)
        childState = self.game.changePerspective(childState)
        child = Node(self.game, self.args, childState, childBoard, self, action)
        self.children.append(child)
        return child
    
    def simulate(self):
        value, isTerminal = self.game.getValAndTerminate(self.board)
        value *= -1
        if isTerminal:
            return value
        
        rolloutState = self.state.copy()
        rolloutBoard = self.board.copy()
        rolloutPlayer = 1
        while True:
            validMoves = self.game.getValidMoves(rolloutBoard)
            action = np.random.choice(validMoves)
            rolloutState, rolloutBoard = self.game.getNextState(rolloutState, action, rolloutBoard)
            value, isTerminal = self.game.getValAndTerminate(rolloutBoard)
            if isTerminal:
                if rolloutPlayer == -1:
                    value *= -1
                return value
            rolloutPlayer *= -1
        
    def backpropogate(self, value):
        self.valueSum += value
        self.visitCount += 1

        value *= -1
        if self.parent is not None:
            self.parent.backpropogate(value)


class MCTS:
    def __init__(self, game: ChessGame, args):
        self.game = game
        self.args = args

    def search(self, state, board):
        root = Node(self.game, self.args, state, board)
        for search in range(self.args['num_searches']):
            node = root
            while node.isFullyExpanded():
                node = node.select()

            value, isTerminal = self.game.getValAndTerminate(node.board)
            value = value * -1 

            if not isTerminal:
                node = node.expand()
                value = node.simulate()

            node.backpropogate(value)
        
        actionProbs = np.zeros(self.game.actionSize)
        for child in root.children:
            actionProbs[child.actionTaken] = child.visitCount
        actionProbs /= np.sum(actionProbs)
        return actionProbs
           
            #backprop
    #return visit counts

