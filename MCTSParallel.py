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
    def search(self, states, boards, idx, spGames):
        policy, _ = self.model(torch.tensor(self.game.getEncodedState(states), device=self.args["device"]))
        policy = torch.softmax(policy, axis=1).cpu().numpy()
        
        root = Node(self.game, self.args, states, boards, prior = 1)
        self.drawer = Drawer(root)

        for i, spg in enumerate(spGames):
            spgPolicy = policy[i]
            validMoves = self.game.getValidMoves(boards[i])
            zeros = np.zeros(4096)
            for move in validMoves:
                zeros[encode(str(move))] = 1
            if boards[i].turn == chess.BLACK:
                zeros = np.transpose(zeros)
            spgPolicy *= zeros
            spgPolicy /= np.sum(spgPolicy)

            spg.root = Node(self.game, self.args, states[i], boards[i], prior = 1)
            spg.root.expand(spgPolicy, spg.root.visitCount+1)

           
            
        for _ in (num_searches := tqdm(range(self.args['num_searches']), leave=False)):
            num_searches.set_description(f"Search {idx}")
            for spg in spGames:
                spg.node = None
                node = spg.root
      
                while node.isFullyExpanded():
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
                zeros = np.zeros(4096)
                for move in validMoves:
                    zeros[encode(str(move))] = 1
                if node.board.turn == chess.BLACK:
                    zeros = np.transpose(zeros)
                spgPolicy *= zeros
                spgPolicy /= np.sum(policy)
                node = node.expand(spgPolicy, spGames[mappingIndx].root.visitCount + 1)
                node.backpropogate(spgValue, spGames[mappingIndx].root.visitCount + 1)
        # self.drawer.update(spGames[0].root)

            

        # self.drawer.update()
        
           
            #backprop
    #return visit counts