import math
from ChessGame import ChessGame
import numpy as np
import chess
from NN import ResNet
import torch
from PrettyPrint import PrettyPrintTree
from tqdm import tqdm

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
device = 'cuda'
def encode(inp: str):
    action = str(inp)
    rowFrom = (int(action[1]) - 1) 
    colFrom = letters.index(action[0]) * 8
    rowWhere = (int(action[3]) - 1) * 64
    colWhere = letters.index(action[2]) * 512 
    return rowFrom + rowWhere + colFrom + colWhere

class Node:
    _ucb = 0

    @property
    def ucb(self):
        return self._ucb

    @ucb.setter
    def ucb(self, value):
        self._ucb = value


    def updateUCB(self, idx, search):
        if self.lastUpdate != search:
            self.ucb = self.getUCB_Self()
            self.lastUpdate = search

        if (not self.parent is None):
            if idx > 0:
                self.parent.children[idx - 1].ucb = self.parent.children[idx - 1].getUCB_Self()
                self.parent.children[idx - 1].lastUpdate = search
                if self.parent.children[idx - 1].ucb < self._ucb:
                    self.parent.children[idx - 1], self.parent.children[idx] = self.parent.children[idx], self.parent.children[idx - 1]
                self.parent.children[idx-1].updateUCB(idx - 1, search)

    def updateUCBFullyVisited(self, idx, search):
        if self.lastUpdate == search:
            return
        self._ucb = self.getUCB_Self()
        self.lastUpdate = search
        if (not self.parent is None):
            while idx < len(self.parent.children)-1:
                value = self.parent.children[idx+1].getUCB_Self()

                if (value > self._ucb):
                    self.parent.children[idx + 1].updateUCBFullyVisited(idx+1, search)
                    self.parent.children[idx], self.parent.children[idx+1] = self.parent.children[idx+1], self.parent.children[idx]
                    idx += 1
                else:
                    self.parent.children[idx+1].ucb = value
                    self.parent.children[idx+1].lastSearch = search
                    
                    break

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
        self.lastUpdate = -1


        self.children = []
        self.noVisitsChildStart = 0

        self.visitCount = 0
        self.valueSum = 0


    
    @property
    def val(self):
        try:
            v = self.valueSum.item()
        except:
            v = self.valueSum
        return f"{str(self.board)}\nVisits: {self.visitCount}\nValue: {v:.3f}\nutc: {self.ucb:.3f}\n" #TODO

    def isFullyExpanded(self):
        return len(self.children) > 0

    def select(self):
        return self.children[0]
        bestChild = None
        bestUCB = -np.inf
        zeroUcb = self.getUCB(self.children[0])
        for child in self.children:
            ucb = self.getUCB(child)
            if ucb > bestUCB:
                bestChild = child
                bestUCB = ucb

        assert newRes == bestChild, f"new UCB = {newRes.ucb}, best UCB = {bestUCB}, old ucb for 0: {zeroUcb}\nold way child idx = {self.children.index(bestChild)}"
        return bestChild
    
    def getUCB(self, child):
        if child.visitCount == 0:
            qValue = 0
        else:
            # qValue = 1 - (child.valueSum / child.visitCount + 1) / 2
            qValue = child.valueSum / child.visitCount
        return qValue + self.args['C'] * ((math.sqrt(self.visitCount) / (child.visitCount + 1))) * child.prior

    def getUCB_Self(self):
        if self.visitCount == 0:
            qValue = 0
        else:
            # qValue = 1 - (self.valueSum / self.visitCount + 1) / 2
            qValue = self.valueSum / self.visitCount
        return qValue + self.args['C'] * ((math.sqrt(self.parent.visitCount+1) / (self.visitCount + 1))) * self.prior


    def  expand(self, policy, search):
        for action, prob in enumerate(policy):
            if prob > 0:
                childState = self.state.copy()
                childBoard = self.board.copy()
                childState, childBoard  = self.game.getNextState(childState, action, childBoard)
                childState = self.game.changePerspective(childState)
                child = Node(self.game, self.args, childState, childBoard, self, action, prob)
                self.children.append(child)
                child.updateUCB(len(self.children), search)

        assert not any([self.children[i].ucb < self.children[i+1].ucb for i in range(len(self.children)-1)])
        return self
    
        
    def backpropogate(self, value, search):
        self.valueSum += value
        self.visitCount += 1

        value *= -1
        if self.parent is not None:
            noVisitsExist = self.parent.noVisitsChildStart < len(self.parent.children)
            if noVisitsExist:
                self.parent.children[self.parent.noVisitsChildStart].updateUCB(self.parent.noVisitsChildStart, search)
            else:
                self.updateUCBFullyVisited(0, search)
            if noVisitsExist and self.parent.children[self.parent.noVisitsChildStart].visitCount != 0:
                self.parent.noVisitsChildStart += 1
            self.parent.backpropogate(value, search)

class Drawer:
    def __init__(self, root: Node):
        self.xStep = 5
        self.yStep = 5
        self.maxX = 0
        self.minY = 0
        self.texts = []
        self.root = root

    def update(self, node):
        
        pt = PrettyPrintTree(lambda x: [y for y in x.children], lambda x: x.val, max_depth=-1, return_instead_of_print=True, color=None)
        tree_as_str = pt(node)
        # with open('tree.txt', 'w', encoding="utf8") as f:
        #     f.write(tree_as_str)
        from PIL import Image
        from PIL import ImageDraw
        from PIL import ImageFont
        img = Image.new('RGB', (100, 100))
        d = ImageDraw.Draw(img)
        fontSize = 20 
        font = ImageFont.truetype("FreeMono.ttf", size=fontSize)

        _, _, width, height = d.textbbox(text=tree_as_str, font=font, xy=(0,0))
        img = Image.new('RGB', (width+2*fontSize, height+2*fontSize))
        d = ImageDraw.Draw(img)
        d.text((20, 20), tree_as_str, fill=(255, 255, 255), font=font)
        img.show()
        img.save('tree.tiff')
        # exit()

        
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

            
        for iter in (num_searches := tqdm(range(self.args['num_searches']), leave=False)):
            num_searches.set_description(f"Searches {idx}")
            node = root
            while node.isFullyExpanded():
                node = node.select()
            # self.drawer.update(root)

            value, isTerminal = self.game.getValAndTerminate(node.board)
            value = value * -1 
            if not isTerminal:
                policy, value = self.model(torch.tensor(self.game.getEncodedState(node.state), device = self.args["device"]).unsqueeze(0))
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                validMoves = self.game.getValidMoves(node.board)
                zeros = np.zeros(4096)
                for move in validMoves:
                    zeros[encode(str(move))] = 1
                if node.board.turn == chess.BLACK:
                    zeros = np.transpose(zeros)
                policy *= zeros
                policy /= np.sum(policy)

                value = value.item()

                node = node.expand(policy, iter)

            node.backpropogate(value, iter)

        # self.drawer.update(root)
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

