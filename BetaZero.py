
import random
import numpy as np
from MCTS import MCTS
import torch
from ChessGame import ChessGame
import torch.nn.functional as F

class BetaZero:
    def __init__(self, model, optimizer, game: ChessGame, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(model, game, args)

    def selfPlay(self):
        memory = []
        player = True
        state, board = self.game.getInitialState()

        state = state * -1
        isTerminal = False
        while True:
            print(board)
            neutralState = state * -1
            actionProbs = self.mcts.search(neutralState, board)
            if not isTerminal:
                memory.append((neutralState, actionProbs))
                action = np.random.choice(self.game.actionSize, p=actionProbs)
                state, board = self.game.getNextState(state, action, board)
                value, isTerminal = self.game.getValAndTerminate(board)
            if isTerminal:
                returnMemory = []
                for idx, tup in enumerate(memory):
                    histNeutralState, histActionProbs = tup
                    histOutcome = value if idx % 2 == 0 else -value
                    returnMemory.append((self.game.getEncodedState(histNeutralState), histActionProbs, histOutcome))
                return returnMemory
            player = not player


    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batchSize']):
            sample = memory[batchIdx:min(len(memory)-1, batchIdx+self.args['batchSize'])]
            state, policyTargets, valueTargets = zip(*sample)
            state, policyTargets, valueTargets = np.array(state), np.array(policyTargets), np.array(valueTargets).reshape(-1, 1)
            state = torch.tensor(state, dtype = torch.float32)
            policyTargets = torch.tensor(policyTargets, dtype = torch.float32)
            valueTargets = torch.tensor(valueTargets, dtype = torch.float32)
            outPolicy, outValue = self.model(state)
            policyLoss = F.cross_entropy(outPolicy, policyTargets)
            valueLoss = F.mse_loss(outValue, valueTargets)
            loss = policyLoss + valueLoss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def learn(self):
        for iteration in range(self.args['numIterations']):
            memory = []

            self.model.eval()
            for selfPlayIteration in range(self.args['numSelfPlayIterations']):
                memory += self.selfPlay()

            self.model.train()
            for _ in range(self.args['numEpochs']):
                self.train(memory)
            torch.save(self.model.state_dict(), f"model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}.pt")