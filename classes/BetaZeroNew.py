import math
import random
import time
import os
import chess
import numpy as np
from numpy.typing import NDArray
import torch
import multiprocessing as mp
from torch.utils.tensorboard.writer import SummaryWriter
import torch.nn.functional as F
from tqdm import tqdm
from classes.ChessGame import ChessGame
from classes.NN import ResNet
from classes.Node import Node
from classes.SPG import SPG
from helpers.chessBoard import getEncodedState, getNextState, getValAndTerminate, getValidMoves


class BetaZero:
    def __init__(self, model: ResNet, optimizer: torch.optim.Optimizer, args: dict) -> None:
        self.model = model
        self.optimizer = optimizer
        # self.game = game
        self.args = args

        Node.C = self.args["C"]

        self.writer = None
        if "track" in self.args and self.args["track"] is True:
            self.writer = SummaryWriter(f"logdir/{time.time()}")

        self.spGames: list[SPG] = []
        self.memory: list[tuple[NDArray[np.floating], NDArray[np.floating], float]] = []

        self.poolSize = 1
        if "numProcesses" in args and args["numProcesses"] > 1:
            self.poolSize: int = min(args["numProcesses"], args["numParallelGames"])
            self.poolSize: int = min(self.poolSize, os.cpu_count())  # type: ignore

    def learn(self):
        numIterations = tqdm(range(self.args["numIterations"]), leave=False)
        numIterations.set_description("Iterations")
        for iteration in numIterations:
            memory = []
            selfPlayIterations = tqdm(range(self.args["numSelfPlayIterations"]), leave=False)
            selfPlayIterations.set_description("Self plays")

            self.spGames: list[SPG] = [SPG(ChessGame()) for _ in range(self.args["numParallelGames"])]

            for _ in selfPlayIterations:
                memory += self.selfPlay()

            self.model.train()
            lossValue, lossPrior = 0, 0
            for _ in (numEpochs := tqdm(range(self.args["numEpochs"]), leave=False)):
                numEpochs.set_description("Epochs")
                lossValueIter, lossPriorIter = self.train(memory)
                lossPrior += lossPriorIter
                lossValue += lossValueIter

            self.updatePlots(lossPrior, lossValue, iteration)
            if "saveFrequency" not in self.args or iteration % self.args["saveFrequency"] == 0:
                torch.save(self.model.state_dict(), f"model_{iteration}.pt")
                torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}.pt")

    def selfPlay(self):
        idx = 0
        player = True
        mates = 0
        self.model.eval()

        self.initRoots()

        while len(self.spGames) > 0:
            tqdm.write(
                f"{idx+1} turn\nRemaining Games {len(self.spGames)}/{self.args['numParallelGames']}\n"
                f"mates: {mates}/{self.args['numParallelGames']-len(self.spGames)}"
            )
            firstBoards = [str(game.board).split("\n") for game in self.spGames[:10]]
            boardStates = "\n".join("".join([f"{el:20}" for el in row]) for row in zip(*firstBoards))
            tqdm.write(f"{boardStates}\n")

            states = np.stack([spg.state for spg in self.spGames])
            boards = [spg.board for spg in self.spGames]

            if self.args["numParallelGames"] > 1:
                actions = []
                poolSize = min(self.poolSize, len(self.spGames))
                step = math.ceil(len(self.spGames) / poolSize)
                # TODO: Проверить как работает пул
                # нужно ли перпесоздавать его на каждый ход или поддерживать старый
                with mp.Pool(poolSize) as pool:
                    for result in pool.map(
                        self.mcts.search,
                        [
                            (states[i : i + step], boards[i : i + step], idx, self.spGames[i : i + step])
                            for i in range(0, len(self.spGames), step)
                        ],
                    ):
                        actions += result
            else:
                actions = [self.mcts.search((states[0], boards[0], idx, [self.spGames[0]]))]

            # TODO: вынести в отдельдую функцию
            for i in range(len(self.spGames)):
                action = np.random.choice(ChessGame.actionSize, p=actions[i][1])
                self.spGames[i].memory.append((actions[i][0], actions[i][1], action, actions[i][3]))
                self.spGames[i].state, self.spGames[i].board = getNextState(
                    self.spGames[i].state, action, board := self.spGames[i].board, board.turn == chess.WHITE
                )
                value, isTerminal = getValAndTerminate(board)
                if "max_game_length" in self.args and self.args["max_game_length"] == idx and not isTerminal:
                    value = 0
                    isTerminal = True
                elif isTerminal:
                    self.gameIsTerminal(self.spGames[i], value, mates)
                    del self.spGames[i]
                else:
                    # TODO: понять делается на доске ход
                    # self.spGames[i].state = changePerspective(self.spGames[i].state)
                    # ? Можно ли привязаться к 0 инлексу при выборе для эвал игр и игр с человеком
                    # ? при этом оставив случайное действие для обучения
                    self.spGames[i].root = [
                        x for x in self.spGames[i].root.children if x.board == self.spGames[i].board
                    ][0]
                player = not player
                idx += 1
        tqdm.write(f"Results: {mates}/{self.args['numParallelGames']}\n")
        return self.memory

    def initRoots(self):
        states = np.stack([spg.state for spg in self.spGames])
        boards = [spg.board for spg in self.spGames]

        policy, _ = self.model(torch.tensor(getEncodedState(states), device=self.args["device"]))
        policy = torch.softmax(policy, axis=1).cpu().detach().numpy()  # type:ignore

        for spg, state, board, spgPolicy in zip(self.spGames, states, boards, policy):
            spgPolicy = (1 - self.args["dirichlet_epsilon"]) * spgPolicy
            spgPolicy += self.args["dirichlet_epsilon"] * np.random.dirichlet(
                [self.args["dirichlet_alpha"]] * ChessGame.actionSize
            )
            validMoves = getValidMoves(board)
            mask = np.where(np.isin(np.arange(ChessGame.actionSize), validMoves), 1, 0)
            if board.turn == chess.BLACK:
                mask = np.flip(mask)
            spgPolicy *= mask
            spSum = np.sum(spgPolicy)
            if spSum > 0:
                spgPolicy /= spSum
            else:
                spgPolicy = mask
            spg.root = Node(state, board, visitCount=1, prior=1)

    def gameIsTerminal(
        self,
        spg: SPG,
        value: float,
        mates: int,
    ):
        # ? А сюда могут оба знака попасть?
        if value == 1 or value == -1:
            mates += 1
        memoryLen = math.ceil(len(spg.memory) / 2)
        for turn, tup in enumerate(spg.memory):
            histNeutralState, histActionProbs, action, mask = tup
            histOutcome = value if turn % 2 == 0 else -value

            if histOutcome < 1:
                base = histActionProbs[action]
                coef = 1 if histOutcome == -1 else 0.5
                nonZero = np.count_nonzero(mask)
                maskNonZero = np.count_nonzero(mask)

                for act, isVal in enumerate(mask):
                    if isVal == 0:
                        continue
                    if act == action:
                        histActionProbs[act] -= ((turn // 2 + 1) / memoryLen) * coef * base / nonZero
                        if histActionProbs[act] < 0:
                            histActionProbs[act] = 0
                    elif histActionProbs[act] != 0:
                        histActionProbs[act] += ((turn // 2 + 1) / memoryLen) * coef * base / nonZero
                    else:
                        histActionProbs[act] += (
                            ((turn // 2 + 1) / memoryLen) * coef * (1 / (maskNonZero - nonZero)) * base
                        )
            else:
                histActionProbs[action] += (turn // 2 + 1) / memoryLen

            actionSum = np.sum(histActionProbs)
            if actionSum > 0:
                histActionProbs /= np.sum(histActionProbs)
            else:
                histActionProbs = mask

            self.memory.append((getEncodedState(histNeutralState), histActionProbs, histOutcome))

    def updatePlots(self, lossPrior: float, lossValue: float, iteration: int):
        if not self.writer:
            return
        lossPrior /= self.args["numEpochs"]
        lossValue /= self.args["numEpochs"]
        self.writer.add_scalar("LossValue/train", lossValue, iteration)
        self.writer.add_scalar("LossPrior/train", lossPrior, iteration)

        # if "eval" in self.args and self.args["eval"] is True:
        #     self.ev()
        #     lossEval = self.train(memory, train=False)
        #     writer.add_scalar("LossEval/test", lossEval, iteration)

    def train(
        self, memory: list[tuple[NDArray[np.floating], NDArray[np.floating], float]], train: bool = True
    ) -> tuple[float, float]:
        # TODO: all to np arrays?
        random.shuffle(memory)
        valLoss = 0
        priorLoss = 0
        for batchIdx in range(0, len(memory), self.args["batchSize"]):
            sample = memory[batchIdx : min(len(memory), batchIdx + self.args["batchSize"])]
            state, policyTargets, valueTargets = zip(*sample)
            state, policyTargets, valueTargets = (
                np.array(state),
                np.array(policyTargets),
                np.array(valueTargets).reshape(-1, 1),
            )
            # TODO: can torch recieve lists?
            state = torch.tensor(state, dtype=torch.float64).to(f'{self.args["device"]}')
            policyTargets = torch.tensor(policyTargets, dtype=torch.float64).to(f'{self.args["device"]}')
            valueTargets = torch.tensor(valueTargets, dtype=torch.float64).to(f'{self.args["device"]}')
            outPolicy, outValue = self.model(state)
            policyLoss = F.cross_entropy(outPolicy, policyTargets)
            valueLoss = F.mse_loss(outValue, valueTargets)
            loss = policyLoss + valueLoss
            valLoss += valueLoss.item()
            priorLoss += policyLoss.item()
            # TODO: why is "if" here?
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return valLoss / (len(memory) / self.args["batchSize"]), priorLoss / (len(memory) / self.args["batchSize"])
