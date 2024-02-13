import math
import random
import time
import os
from typing import List
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
from classes.MCTS import MCTS
from helpers.chessBoard import getEncodedState, getValAndTerminate, getValidMoves


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

        self.mcts = MCTS(self.model, self.args)
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

            self.model.eval()

            for _ in selfPlayIterations:
                memory += self.selfPlay([SPG(ChessGame()) for _ in range(self.args["numParallelGames"])])

            self.model.train()
            lossValue, lossPrior = 0, 0
            for _ in (numEpochs := tqdm(range(self.args["numEpochs"]), leave=False)):
                numEpochs.set_description("Epochs")
                lossValueIter, lossPriorIter = self.train(memory)
                lossPrior += lossPriorIter
                lossValue += lossValueIter

            self.updatePlots(lossPrior, lossValue, iteration)
            if "saveFrequency" not in self.args or iteration % self.args["saveFrequency"] == 0:
                torch.save(self.model.state_dict(), f"{self.args['newModelsDir']}/model_{iteration}.pt")
                torch.save(self.optimizer.state_dict(), f"{self.args['newOptimizersDir']}/optimizer_{iteration}.pt")

    def selfPlay(self, spgs: List[SPG]):
        idx = 0
        player = True
        mates = 0
        self.initRoots(spgs)

        while len(spgs) > 0:
            tqdm.write(
                f"{idx+1} turn\nRemaining Games {len(spgs)}/{self.args['numParallelGames']}\n"
                f"mates: {mates}/{self.args['numParallelGames']-len(spgs)}"
            )
            firstBoards = [str(game.board).split("\n") for game in spgs[:10]]
            boardStates = "\n".join("".join([f"{el:20}" for el in row]) for row in zip(*firstBoards))
            tqdm.write(f"{boardStates}\n")

            states = np.stack([spg.state for spg in spgs])

            if self.args["numParallelGames"] > 1:
                actions = []
                poolSize = min(self.poolSize, len(spgs))
                step = math.ceil(len(spgs) / poolSize)
                # TODO: Проверить как работает пул
                # нужно ли перпесоздавать его на каждый ход или поддерживать старый
                # statesPool = [states[i : i + step] for i in range(0, len(self.spGames), step)]
                # spGamesPool = [self.spGames[i : i + step] for i in range(0, len(self.spGames), step)]

                with mp.Pool(poolSize) as pool:
                    for result, updateSpGames in pool.starmap(
                        self.mcts.search,
                        [(states[i : i + step], spgs[i : i + step]) for i in range(0, len(spgs), step)],
                    ):
                        actions += result
                        spgs = updateSpGames
            else:
                actions, spgs = self.mcts.search(states, spgs)

            # TODO: вынести в отдельдую функцию
            for i in range(len(spgs) - 1, -1, -1):
                # TODO: What is actually returned and why do we need so much nested lists?
                action = np.random.choice(ChessGame.actionSize, p=actions[i][1])
                spgs[i].memory.append((actions[i][0], actions[i][1], action, actions[i][3]))
                spgs[i].root = spgs[i].root.find(action)  # type: ignore
                # spgs[i].state, spgs[i].board = getNextState(
                #     spgs[i].state, action, board := spgs[i].board, board.turn == chess.WHITE
                # )
                value, isTerminal = getValAndTerminate(spgs[i].root.board)  # type: ignore
                if "max_game_length" in self.args and self.args["max_game_length"] - 1 <= idx and not isTerminal:
                    value = 0
                    isTerminal = True

                if isTerminal:
                    # ? А сюда могут оба знака попасть?
                    if value == 1 or value == -1:
                        mates += 1
                    # ? Как это вообще работает?
                    self.gameIsTerminal(spgs[i], value)
                    del spgs[i]
                else:
                    # TODO: понять делается ли на доске ход
                    # TODO: походу состояние дерева не обновляется
                    # self.spGames[i].state = changePerspective(self.spGames[i].state)
                    # ? Можно ли привязаться к 0 инлексу при выборе для эвал игр и игр с человеком
                    # ? при этом оставив случайное действие для обучения
                    # spgs[i].root = spgs[i].node
                    # if len(self.spGames[i].root.children) == 0:
                    #     self.spGames[i].root = self.spGames[i].root.expand()

                    # if len(self.spGames[i].root.children) > 0:
                    #     # self.spGames[i].root = self.spGames[i].root.children[0]
                    #     self.spGames[i].root = self.spGames[i].root.find(action)
                    #     # TODO: переписать функцию чтобы не копировать все это
                    #     if len(self.spGames[i].root.children) == 0:
                    #         state = self.spGames[i].root.state
                    #         board = self.spGames[i].root.board

                    #         policy, _ = self.model(
                    #             torch.tensor(np.squeeze(getEncodedState(state), axis=1), device=self.args["device"])
                    #         )
                    #         policy = torch.softmax(policy, axis=1).cpu().detach().numpy()  # type:ignore

                    #         spgPolicy = (1 - self.args["dirichlet_epsilon"]) * policy
                    #         spgPolicy += self.args["dirichlet_epsilon"] * np.random.dirichlet(
                    #             [self.args["dirichlet_alpha"]] * ChessGame.actionSize
                    #         )
                    #         validMoves = getValidMoves(board)
                    #         mask = np.where(np.isin(np.arange(ChessGame.actionSize), validMoves), 1, 0)
                    #         if board.turn == chess.BLACK:
                    #             mask = np.flip(mask)
                    #         spgPolicy *= mask
                    #         spSum = np.sum(spgPolicy)
                    #         if spSum > 0:
                    #             spgPolicy /= spSum
                    #         else:
                    #             spgPolicy = mask
                    #         self.spGames[i].root.expand(spgPolicy, self.spGames[i].root.board.turn == chess.WHITE)
                    #     else:
                    #         self.spGames[i].root = self.spGames[i].root.noChildren[0]
                    # selected = [x for x in self.spGames[i].root.children if x.board == self.spGames[i].board]

                    # if len(selected) > 0:
                    #     self.spGames[i].root = selected[0]
                    # else:
                    #     self.spGames[i].root = [
                    #         x for x in self.spGames[i].root.noChildren if x.board == self.spGames[i].board
                    #     ][0]
                    player = not player
                    idx += 1
        tqdm.write(f"Results: {mates}/{self.args['numParallelGames']}\n")
        return self.memory

    def initRoots(self, spgs: List[SPG]):
        states = np.stack([spg.state for spg in spgs])
        boards = [spg.board for spg in spgs]

        policy, _ = self.model(torch.tensor(getEncodedState(states), device=self.args["device"]))
        policy = torch.softmax(policy, axis=1).cpu().detach().numpy()  # type:ignore

        for spg, state, board, spgPolicy in zip(spgs, states, boards, policy):
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
            spg.root.expand(spgPolicy, spg.root.board.turn == chess.WHITE)

    def gameIsTerminal(
        self,
        spg: SPG,
        value: float,
    ):
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
            state = torch.tensor(state, dtype=torch.float32).to(f'{self.args["device"]}')
            policyTargets = torch.tensor(policyTargets, dtype=torch.float32).to(f'{self.args["device"]}')
            valueTargets = torch.tensor(valueTargets, dtype=torch.float32).to(f'{self.args["device"]}')
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
