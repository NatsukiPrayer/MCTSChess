import random
import chess
import numpy as np
from MCTS import MCTS
import torch
from ChessGame import ChessGame
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F
import time as t

device = "cuda"


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
        isTerminal = False
        idx = 0
        state = self.game.changePerspective(state)
        tqdm.write("New game started\n")
        while True:
            tqdm.write(f"{str(board)}\n")
            neutralState = self.game.changePerspective(state)
            actionProbs = self.mcts.search(neutralState, board, idx)
            if not isTerminal:
                memory.append((neutralState, actionProbs))
                action = np.random.choice(self.game.actionSize, p=actionProbs)
                state, board = self.game.getNextState(neutralState, action, board, board.turn == chess.WHITE)
                value, isTerminal = self.game.getValAndTerminate(board)
                if idx > 30:
                    value = 0
                    isTerminal = True

            if isTerminal:
                returnMemory = []
                for idx, tup in enumerate(memory):
                    histNeutralState, histActionProbs = tup
                    histOutcome = value if idx % 2 == 0 else -value  # type: ignore
                    returnMemory.append(
                        (
                            self.game.getEncodedState(histNeutralState),
                            histActionProbs,
                            histOutcome,
                        )
                    )
                return returnMemory
            player = not player
            idx += 1

    def train(self, memory, train=True):
        random.shuffle(memory)
        train_loss = 0
        for batchIdx in range(0, len(memory), self.args["batchSize"]):
            sample = memory[batchIdx : min(len(memory), batchIdx + self.args["batchSize"])]
            state, policyTargets, valueTargets = zip(*sample)
            state, policyTargets, valueTargets = (
                np.array(state),
                np.array(policyTargets),
                np.array(valueTargets).reshape(-1, 1),
            )
            state = torch.tensor(state, dtype=torch.float32).to(f"{device}")
            policyTargets = torch.tensor(policyTargets, dtype=torch.float32).to(f"{device}")
            valueTargets = torch.tensor(valueTargets, dtype=torch.float32).to(f"{device}")
            outPolicy, outValue = self.model(state)
            policyLoss = F.cross_entropy(outPolicy, policyTargets)
            valueLoss = F.mse_loss(outValue, valueTargets)
            loss = policyLoss + valueLoss
            train_loss += loss.item()
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return train_loss / (len(memory) / self.args["batchSize"])

    def learn(self):
        writer = SummaryWriter(f"logdir/{t.time()}")
        for iteration in (numIterations := tqdm(range(self.args["numIterations"]), leave=False)):
            numIterations.set_description("Iterations")
            memory = []

            self.model.eval()
            for selfPlayIteration in (
                numSelfPlayIterations := tqdm(range(self.args["numSelfPlayIterations"]), leave=False)
            ):
                numSelfPlayIterations.set_description("Self plays")
                memory += self.selfPlay()

            self.model.train()

            for epoch in (numEpochs := tqdm(range(self.args["numEpochs"]), leave=False)):
                numEpochs.set_description("Epochs")
                loss = self.train(memory)
                self.ev()
                lossTest = self.train(memory, train=False)
                writer.add_scalar("Loss/train", loss, epoch)
                writer.add_scalar("Loss/test", lossTest, epoch)
            torch.save(self.model.state_dict(), f"model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}.pt")

    @torch.no_grad()
    def ev(self):
        self.model.eval()

    def learn2(self, train, test):
        writer = SummaryWriter(f"logdir/{t.time()}")
        for epoch in range(self.args["numEpochs"]):
            self.model.train()
            loss = self.train(train)
            self.ev()
            lossTest = self.train(train, train=False)
            writer.add_scalar("Loss/train", loss, epoch)
            writer.add_scalar("Loss/test", lossTest, epoch)
            torch.save(self.model.state_dict(), f"model_{epoch}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{epoch}.pt")
        writer.close()
