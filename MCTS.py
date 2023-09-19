import math
from ChessGame import ChessGame
import numpy as np
import chess
from NN import ResNet
import torch
from PrettyPrint import PrettyPrintTree
from tqdm import tqdm

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

def encode(inp: str):
    action = str(inp)
    rowFrom = (int(action[1]) - 1) 
    colFrom = letters.index(action[0]) * 8
    rowWhere = (int(action[3]) - 1) * 64
    colWhere = letters.index(action[2]) * 512 
    return rowFrom + rowWhere + colFrom + colWhere

class Node:
    def __init__(self, game: ChessGame, args, state, board: chess.Board, parent=None, actionTaken = None, prior=0):
        self.game = game
        self.args = args
        self.state = state
        self.board = board
        self.parent = parent
        

        if isinstance(actionTaken, int) or actionTaken is None:
            self.actionTaken = actionTaken
        else:
            self.actionTaken = encode(actionTaken)
        self.prior = prior


        self.children = []

        self.visitCount = 0
        self.valueSum = 0
    
    @property
    def val(self):
        return f"{str(self.board)}\nVisits: {self.visitCount}\nValue: {self.valueSum}"

    def isFullyExpanded(self):
        return len(self.children) > 0

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
        if child.visitCount == 0:
            qValue = 0
        else:
            qValue = 1 - (child.valueSum / child.visitCount + 1) / 2
        return qValue + self.args['C'] * ((math.sqrt(self.visitCount) / (child.visitCount + 1))) * child.prior


    def  expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                childState = self.state.copy()
                childBoard = self.board.copy()
                childState, childBoard  = self.game.getNextState(childState, action, childBoard)
                childState = self.game.changePerspective(childState)
                child = Node(self.game, self.args, childState, childBoard, self, action, prob)
                self.children.append(child)
        return self
    
        
    def backpropogate(self, value):
        self.valueSum += value
        self.visitCount += 1

        value *= -1
        if self.parent is not None:
            self.parent.backpropogate(value)

class Drawer:
    def __init__(self, root: Node):
        self.xStep = 5
        self.yStep = -5
        self.maxX = 0
        self.minY = 0
        self.root = root

    def update(self, x: float = 0.5, y: float = 0.5):
        pt = PrettyPrintTree(lambda x: x.children, lambda x: x.val, max_depth=2, return_instead_of_print=True)
        tree_as_str = pt(self.root)
        with open('tree.txt', 'w', encoding="utf-8") as f:
            f.write(tree_as_str.encode("utf-8").decode('utf-8'))

        
class MCTS:
    def __init__(self, model, game: ChessGame, args):
        self.game = game
        self.args = args
        self.model = model
        self.drawer = None

    @torch.no_grad()
    def search(self, state, board, idx):
        root = Node(self.game, self.args, state, board, prior = 1)
        self.drawer = Drawer(root)

            
        for _ in (num_searches := tqdm(range(self.args['num_searches']), leave=False)):
            # self.drawer.update()
            num_searches.set_description(f"Searches {idx}")
            node = root
            while node.isFullyExpanded():
                node = node.select()

            value, isTerminal = self.game.getValAndTerminate(node.board)
            value = value * -1 

            if not isTerminal:
                policy, value = self.model(torch.tensor(self.game.getEncodedState(node.state)).unsqueeze(0))
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                validMoves = self.game.getValidMoves(node.board)
                zeros = np.zeros(4096)
                for move in validMoves:
                    zeros[encode(str(move))] = 1
                policy *= zeros
                policy /= np.sum(policy)

                value = value.item()

                node = node.expand(policy)

            node.backpropogate(value)

        # self.drawer.update()
        actionProbs = np.zeros(self.game.actionSize)
        if len(root.children) == 0:
            raise Exception("GameEnd???")
        for child in root.children:
            actionProbs[child.actionTaken] = child.visitCount
        a = np.sum(actionProbs)
        actionProbs /= a
        return actionProbs
           
            #backprop
    #return visit counts

