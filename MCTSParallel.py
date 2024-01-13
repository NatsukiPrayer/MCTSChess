from ChessGame import ChessGame
from MCTS import Node, Drawer, encode
import torch
from tqdm import tqdm
import chess
import numpy as np


class MCTSParallel:
    def __init__(self, model, game: ChessGame, args):
        self.game = game
        self.args = args
        self.model = model
        self.drawer = None


    @torch.no_grad()
    def search(self, *args):
        states, boards, idx, spGames = args[0]
        policy, _ = self.model(torch.tensor(self.game.getEncodedState(states), device=self.args["device"]))
        policy = torch.softmax(policy, axis=1).cpu().numpy()
        
        # root = Node(self.game, self.args, states, boards, prior = 1)

        for i, spg in enumerate(spGames):
            spgPolicy = policy[i]
            spgPolicy = (1-self.args["dirichlet_epsilon"])*spgPolicy+self.args["dirichlet_epsilon"]\
                *np.random.dirichlet([self.args["dirichlet_alpha"]] * self.game.actionSize)
            validMoves = self.game.getValidMoves(boards[i])
            mask = np.zeros(4096)
            for move in validMoves:
                mask[encode(str(move))] = 1
            if boards[i].turn == chess.BLACK:
                mask = np.flip(mask)
            spgPolicy *= mask
            spSum = np.sum(spgPolicy)
            if spSum > 0:
                spgPolicy /= spSum
            else:
                spgPolicy = mask

            spg.root = Node(self.game, self.args, states[i], boards[i], visitCount=1, prior = 1)
            spg.root.expand(spgPolicy, mask, spg.root.board.turn == chess.WHITE)

           
        self.drawer = Drawer(spGames[0].root)
            
        # for _ in (num_searches := tqdm(range(self.args['num_searches']), leave=False)):
            # num_searches.set_description(f"Search {idx}")
        for _ in range(self.args['num_searches']):
            for spg in spGames:
                spg.node = None
                node = spg.root
      
                while node.isFullyExpanded():
                    prevNode = node
                    node = node.select()
            
                value, isTerminal = self.game.getValAndTerminate(node.board)
                value = value * -1 
            
                if isTerminal:
                    node.backpropogate(value, spg.root.visitCount + 1)
                else:
                    spg.node = node
        
            expandableSpgames = [mappingIndx for mappingIndx in range(len(spGames)) if spGames[mappingIndx].node is not None]
           
            if len(expandableSpgames) > 0:
                states = np.stack([spGames[mappingIndx].node.state for mappingIndx in expandableSpgames]) 
                policy, value = self.model(torch.tensor(self.game.getEncodedState(states), device = self.args["device"]))
                policy = torch.softmax(policy, axis=1).cpu().numpy()
                value = value.cpu().numpy()
                boards = [spGames[mappingIndx].node.board for mappingIndx in expandableSpgames]
            for i, mappingIndx in enumerate(expandableSpgames):
                node = spGames[mappingIndx].node
                spgPolicy, spgValue = policy[i], value[i]
                validMoves = self.game.getValidMoves(node.board)
                mask = np.zeros(4096)
                for move in validMoves:
                    mask[encode(str(move))] = 1
                if node.board.turn == chess.BLACK:
                    mask = np.flip(mask)
                spgPolicy *= mask
                sum = np.sum(spgPolicy)
                if sum > 0:
                    spgPolicy /= np.sum(spgPolicy)
                else:
                    spgPolicy = mask                
                node = node.expand(spgPolicy, mask, node.board.turn == chess.WHITE)
                node.backpropogate(spgValue, spGames[mappingIndx].root.visitCount + 1)
        # self.drawer.update(spGames[0].root)
        actions = []
        for spg in spGames:
            actionProbs = np.zeros(self.game.actionSize)
            mask = np.zeros(self.game.actionSize)
            for child in spg.root.children:
                mask[child.actionTaken] = 1
                if child.visitCount != 0:
                    actionProbs[child.actionTaken] = child.visitCount

            actionProbs = actionProbs ** (1/self.args['temperature'])
                
            a = np.sum(actionProbs)
            actionProbs /= a
            spgVal = spg.root.valueSum/spg.root.visitCount
            spgVal = spgVal if spg.root.board.turn else -spgVal
            actions.append((spg.root.state, actionProbs, spgVal, mask))
        # self.drawer.update(spg.root)
        return actions
            #backprop
    #return visit counts