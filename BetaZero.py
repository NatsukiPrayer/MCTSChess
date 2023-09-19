
import random
import numpy as np
from MCTS import MCTS
import torch
from ChessGame import ChessGame
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F
import time as t
device = 'cuda'

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
        print('DOSKA IS REFRESHED')
        state, board = self.game.getInitialState()

        state = state * -1
        isTerminal = False
        idx = 0
        while True:
            # print(board)
            neutralState = state * -1
            actionProbs = self.mcts.search(neutralState, board, idx)
            if not isTerminal:
                memory.append((neutralState, actionProbs))
                action = np.argmax(actionProbs)
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
            idx += 1


    def train(self, memory, train=True):
        random.shuffle(memory)
        train_loss = 0
        for batchIdx in range(0, len(memory), self.args['batchSize']):
            sample = memory[batchIdx:min(len(memory)-1, batchIdx+self.args['batchSize'])]
            state, policyTargets, valueTargets = zip(*sample)
            state, policyTargets, valueTargets = np.array(state), np.array(policyTargets), np.array(valueTargets).reshape(-1, 1)
            state = torch.tensor(state, dtype = torch.float32).to(f'{device}')
            policyTargets = torch.tensor(policyTargets, dtype = torch.float32).to(f'{device}')
            valueTargets = torch.tensor(valueTargets, dtype = torch.float32).to(f'{device}')
            outPolicy, outValue = self.model(state)
            policyLoss = F.cross_entropy(outPolicy, policyTargets)
            valueLoss = F.mse_loss(outValue, valueTargets)
            loss = policyLoss + valueLoss
            train_loss += loss.item()
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return train_loss / (len(memory)/self.args['batchSize'])


    def learn(self):
        for iteration in (numIterations := tqdm(range(self.args['numIterations']), leave=False)):
            numIterations.set_description("Iterations")
            memory = []

            self.model.eval()
            for selfPlayIteration in (numSelfPlayIterations := tqdm(range(self.args['numSelfPlayIterations']), leave=False)):
                numSelfPlayIterations.set_description("Self plays")
                memory += self.selfPlay()

            self.model.train()
            for _ in (numEpochs:= tqdm(range(self.args['numEpochs']), leave=False)):
                numEpochs.set_description("Epochs")
                self.train(memory)
            torch.save(self.model.state_dict(), f"model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}.pt")
    
    @torch.no_grad
    def ev(self):
        self.model.eval()

    def learn2(self, train, test):
        writer = SummaryWriter(f'logdir/{t.time()}')
        for epoch in range(self.args['numEpochs']):
            self.model.train()
            loss = self.train(train)
            self.ev()
            lossTest = self.train(train, train=False)
            writer.add_scalar('Loss/train', loss, epoch)
            writer.add_scalar('Loss/test', lossTest, epoch)
            torch.save(self.model.state_dict(), f"model_{epoch}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{epoch}.pt")
        writer.close()
