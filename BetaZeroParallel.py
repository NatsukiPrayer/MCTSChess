import math
import os
import chess
from MCTSParallel import MCTSParallel
from SPG import SPG

import random
import numpy as np
from MCTS import MCTS, encode
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
        returnMemory = []
        player = True
        spGames = [SPG(self.game) for _ in range(self.args['numParallelGames'])]


        idx = 0
        # for spg in spGames:
        #     spg.state = self.game.changePerspective(spg.state)
        mates = 0
        tqdm.write('New game started\n')
        with mp.Pool(self.poolSize) as pool:
            while len(spGames) > 0:
                # Steps info and first boards print
                tqdm.write(f"{idx+1} turn\nRemaining Games {len(spGames)}/{self.args['numParallelGames']}\n"\
                           f"mates: {mates}/{self.args['numParallelGames']-len(spGames)}")
                firstBoards = [str(game.board).split("\n") for game in spGames[:10]]
                boardStates = '\n'.join(''.join([f"{el:20}" for el in row]) for row in zip(*firstBoards))
                tqdm.write(f"{boardStates}\n")
                states = np.stack([spg.state for spg in spGames])
                
                boards = [spg.board for spg in spGames]
                step = math.ceil(len(spGames) / self.poolSize)
                
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

                    action = np.random.choice(self.game.actionSize, p=actions[i][1])
                    # action = np.argmax(actions[i][1])
                    spg.memory.append((actions[i][0], actions[i][1], action, actions[i][3]))
                    spg.state, spg.board = self.game.getNextState(spg.state, action, spg.board, spg.board.turn == chess.WHITE)
                    value, isTerminal = self.game.getValAndTerminate(spg.board)
                    if idx == 15 and not isTerminal:
                        # value = actions[i][2].item()
                        value = 0
                        isTerminal = True

                    if isTerminal:
                        if value == 1 or value == -1:
                            mates += 1
                        memoryLen = math.ceil(len(spg.memory)/2)
                        for turn, tup in enumerate(spg.memory):
                            histNeutralState, histActionProbs, action, mask = tup
                            histOutcome = value if turn % 2 == 0 else -value

                            if histOutcome < 1:
                                base = histActionProbs[action]
                                coef = 1 if histOutcome == -1 else 0.5
                                nonZero = np.count_nonzero(histActionProbs)
                                maskNonZero = np.count_nonzero(mask)
                                for act, isVal in enumerate(mask):
                                    if isVal == 0:
                                        continue
                                    if act == action:
                                        histActionProbs[act] -= ((turn//2+1)/memoryLen) * coef * base / nonZero
                                        if histActionProbs[act] < 0:
                                            histActionProbs[act] = 0
                                    elif histActionProbs[act] != 0:
                                        histActionProbs[act] += ((turn//2+1)/memoryLen) * coef * base / nonZero
                                    else:
                                        histActionProbs[act] += ((turn//2+1)/memoryLen) * coef * (1 / (maskNonZero - nonZero)) * base
                            else:
                                histActionProbs[action] += ((turn//2+1)/memoryLen)
                            
                            histActionProbs /= np.sum(histActionProbs)

                            # histActionProbs[action] *= histOutcome
                            # histActionProbs[action] += ((turn//2+1)/memoryLen) * (1 if histOutcome > 0 else -1)
                            # if histActionProbs[action] < 0:
                            #     mask = np.zeros(4096)
                            #     for act, prob in enumerate(histActionProbs):
                            #         if prob != 0:
                            #             mask[act] = 1
                            # if np.sum(histActionProbs) <= 0:
                            #     histActionProbs[action] = 0
                            # if np.sum(histActionProbs) > 0:
                            #     histActionProbs[action] /= np.sum(histActionProbs)
                            # else:
                            #     if np.sum(mask) != 1:
                            #         mask[action] = 0
                            #         histActionProbs = mask / np.sum(mask)
                            returnMemory.append((self.game.getEncodedState(histNeutralState), histActionProbs, histOutcome))
                        del spGames[i]
                    else:
                        spg.state = self.game.changePerspective(spg.state)
                player = not player
                idx += 1
        tqdm.write(f"Results: {mates}/{self.args['numParallelGames']}\n")
        return returnMemory


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