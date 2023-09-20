from MCTSParallel import MCTSParallel
from SPG import SPG

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

class BetaZeroParallel:
    def __init__(self, model, optimizer, game: ChessGame, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTSParallel(model, game, args)

    def selfPlay(self):
        returnMemory = []
        player = True
        spGames = [SPG(self.game) for _ in range(self.args['numParallelGames'])]


        idx = 0
        tqdm.write('New game started\n')
        while len(spGames) > 0:
            firstBoards = [str(game.board).split("\n") for game in spGames[:10]]
            boardStates = '\n'.join(''.join([f"{el:20}" for el in row]) for row in zip(*firstBoards))
            tqdm.write(f"{boardStates}\n")
            states = np.stack([spg.state for spg in spGames])
            boards = [spg.board for spg in spGames]
            neutralStates = states * -1
            self.mcts.search(neutralStates, boards, idx, spGames)

            for i in (numGames := tqdm(range(len(spGames))[::-1], leave=False)):
                numGames.set_description(f'Game {idx}')
                spg = spGames[i]

                actionProbs = np.zeros(self.game.actionSize)
                if len(spg.root.children) == 0:
                    raise Exception("GameEnd???")
                for child in spg.root.children:
                    actionProbs[child.actionTaken] = child.visitCount
                a = np.sum(actionProbs)
                actionProbs /= a
                spg.memory.append((spg.root.state, actionProbs))
                action = np.random.choice(self.game.actionSize, p=actionProbs)
                spgState, board = self.game.getNextState(spg.state, action, spg.board)
                value, isTerminal = self.game.getValAndTerminate(spg.board)
                
                
                if isTerminal:
                    returnMemory = []
                    for idx, tup in enumerate(spg.memory):
                        histNeutralState, histActionProbs = tup
                        histOutcome = value if idx % 2 == 0 else -value
                        returnMemory.append((self.game.getEncodedState(histNeutralState), histActionProbs, histOutcome))
                    del spGames[i]
            player = not player
            idx += 1
        return returnMemory


    def train(self, memory, train=True):
        random.shuffle(memory)
        train_loss = 0
        for batchIdx in range(0, len(memory), self.args['batchSize']):
            sample = memory[batchIdx:min(len(memory), batchIdx+self.args['batchSize'])]
            state, policyTargets, valueTargets = zip(*sample)
            state, policyTargets, valueTargets = np.array(state), np.array(policyTargets), np.array(valueTargets).reshape(-1, 1)
            state = torch.tensor(state, dtype = torch.float32).to(f'{self.args["device"]}')
            policyTargets = torch.tensor(policyTargets, dtype = torch.float32).to(f'{self.args["device"]}')
            valueTargets = torch.tensor(valueTargets, dtype = torch.float32).to(f'{self.args["device"]}')
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
    
    @torch.no_grad()
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