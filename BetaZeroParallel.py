import math
import os
import chess
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
import multiprocessing as mp


class BetaZeroParallel:
    def __init__(self, model, optimizer, game: ChessGame, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTSParallel(model, game, args)
        self.poolSize = min(args["numProcesses"], args["numParallelGames"])
        self.poolSize = min(self.poolSize, os.cpu_count())

    def selfPlay(self):
        player = True
        spGames = [SPG(self.game) for _ in range(self.args['numParallelGames'])]


        idx = 0
        # for spg in spGames:
        #     spg.state = self.game.changePerspective(spg.state)
        mates = 0
        tqdm.write('New game started\n')
        with mp.Pool(self.poolSize) as pool:
            while len(spGames) > 0:
                firstBoards = [str(game.board).split("\n") for game in spGames[:10]]
                boardStates = '\n'.join(''.join([f"{el:20}" for el in row]) for row in zip(*firstBoards))
                tqdm.write(f"{boardStates}\n")
                states = np.stack([spg.state for spg in spGames])
                
                boards = [spg.board for spg in spGames]
                step = math.ceil(self.poolSize / len(spGames))
                
                actions = []
                for result in pool.map(self.mcts.search, 
                                [(states[i:i+step], 
                                    boards[i:i+step], 
                                    idx, 
                                    spGames[i:i+step]) for i in range(0, len(spGames), step)]):
                    actions += result

                numGames = tqdm(range(len(spGames))[::-1], leave=False)
                for i in (numGames):
                    numGames.set_description(f'Game {idx}')
                    spg = spGames[i]
                    
                    if spg.board.turn == chess.WHITE:
                        stateCheck = spg.state
                    else:
                        stateCheck = self.game.changePerspective(spg.state)

                    boardState = self.game.posFromFen(spg.board.fen())

                    assert all([all([el1 == el2 for el1, el2 in zip(row1, row2)]) for row1, row2 in zip(boardState, stateCheck)]), "Board and state different"

                    # actionProbs = np.zeros(self.game.actionSize)
                    # if len(spg.root.children) == 0:
                    #     raise Exception("GameEnd???")
                    # for child in spg.root.children:
                    #     if child.visitCount != 0:
                    #         # actionProbs[child.actionTaken] = np.exp(-child.valueSum/child.visitCount)
                    #         actionProbs[child.actionTaken] = child.visitCount
                    # a = np.sum(actionProbs)
                    # actionProbs /= a
                    action = np.random.choice(self.game.actionSize, p=actions[i][1])
                    # action = np.argmax(actionProbs)
                    spg.memory.append((actions[i][0], actions[i][1], action))
                    spg.state, spg.board = self.game.getNextState(spg.state, action, spg.board, spg.board.turn == chess.WHITE)
                    value, isTerminal = self.game.getValAndTerminate(spg.board)
                    if idx == 16:
                        value  = 0
                        isTerminal = True

                    if isTerminal:
                        if value != 0:
                            mates += 1
                        value = -value if spg.board.turn else value
                        for turn, tup in enumerate(spg.memory):
                            histNeutralState, histActionProbs, action = tup
                            histOutcome = value if turn % 2 == 0 else -value
                            mask = np.zeros(self.game.actionSize)
                            for action, prob in enumerate(histActionProbs):
                                if prob > 0:
                                    mask[action] = 1
                            histActionProbs[action] *= histOutcome
                            histActionProbs -= np.min(histActionProbs)
                            histActionProbs *= mask
                            actionSum = np.sum(histActionProbs)
                            if actionSum > 0:
                                histActionProbs /= actionSum
                            else:
                                histActionProbs = mask
                                histActionProbs /= np.sum(histActionProbs)
                            self.game.memory.append((self.game.getEncodedState(histNeutralState), histActionProbs, histOutcome))
                        del spGames[i]
                    else:
                        spg.state = self.game.changePerspective(spg.state)
                player = not player
                idx += 1
        tqdm.write(f"Results: {mates}/{self.args['numParallelGames']}\n")
        return self.game.memory


    def train(self, memory, train=True):
        random.shuffle(memory)
        valLoss = 0
        priorLoss = 0
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
            valLoss += valueLoss.item()
            priorLoss += policyLoss.item()
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return valLoss / (len(memory)/self.args['batchSize']), priorLoss / (len(memory)/self.args['batchSize'])


    def learn(self):
        writer = SummaryWriter(f'logdir/{t.time()}')
        numIterations = tqdm(range(self.args['numIterations']), leave=False)
        numIterations.set_description("Iterations")
        for iteration in (numIterations):
            memory = []

            self.model.eval()
            numSelfPlayIterations = tqdm(range(self.args['numSelfPlayIterations']), leave=False)
            numSelfPlayIterations.set_description("Self plays")
            for selfPlayIteration in (numSelfPlayIterations):
                memory += self.selfPlay()

            self.model.train()
            lossValue, lossPrior = 0,0
            for i in (numEpochs:= tqdm(range(self.args['numEpochs']), leave=False)):
                numEpochs.set_description("Epochs")
                lossValueIt, lossPriorIt = self.train(memory)
                lossPrior += lossPriorIt
                lossValue += lossValueIt
                # self.ev()
                # lossTest = self.train(memory, train=False)
            lossPrior /= self.args["numEpochs"]
            lossValue /= self.args["numEpochs"]
                              
            writer.add_scalar('LossValue/train', lossValue, iteration)
            writer.add_scalar('LossPrior/train', lossPrior, iteration)
                # writer.add_scalar('Loss/test', lossTest, i)
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