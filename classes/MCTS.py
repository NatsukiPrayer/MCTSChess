import torch
import chess
import numpy as np
from numpy.typing import NDArray
from typing import List
from classes.SPG import SPG
from classes.ChessGame import ChessGame
from helpers.chessBoard import getEncodedState, getValAndTerminate, getValidMoves


class MCTS:
    def __init__(self, model, args):
        self.args = args
        self.model = model
        self.drawer = None

    @torch.no_grad()
    def search(self, states: NDArray[np.floating], spGames: List[SPG]):
        policy, _ = self.model(torch.tensor(getEncodedState(states), device=self.args["device"]))
        # TODO: define policy type correctly
        policy = torch.softmax(policy, axis=1).cpu().numpy()  # type:ignore

        # ! TODO: we always stays in root node
        # ? TODO: where root.children should be selected?
        # TODO: can we get rid of self.args?
        for _ in range(self.args["num_searches"]):
            for spg in spGames:
                node = spg.originalRoot

                while node.isFullyExpanded():
                    node = node.select()

                value, isTerminal = getValAndTerminate(node.board)
                value = value * -1

                if isTerminal:
                    node.backpropogate(value)

            # TODO: переписать нормально, зачем mappingIndx?
            expandableSpgames = [
                mappingIndx for mappingIndx in range(len(spGames)) if len(spGames[mappingIndx].root.children) > 0
            ]

            if len(expandableSpgames) > 0:
                # fmt: off
                states = np.stack([spGames[mappingIndx].root.state for mappingIndx in expandableSpgames])  # type:ignore
                # fmt: on
                policy, value = self.model(torch.tensor(getEncodedState(states), device=self.args["device"]))
                policy = torch.softmax(policy, axis=1).cpu().numpy()  # type: ignore
                value = value.cpu().numpy()
                # boards = [spGames[mappingIndx].node.board for mappingIndx in expandableSpgames]

            for i, mappingIndx in enumerate(expandableSpgames):
                node = spGames[mappingIndx].root
                spgPolicy, spgValue = policy[i], value[i]  # type: ignore
                validMoves = getValidMoves(node.board)  # type: ignore
                mask = np.zeros(4096)
                for move in validMoves:
                    mask[move] = 1
                if node.board.turn == chess.BLACK:  # type: ignore
                    mask = np.flip(mask)
                spgPolicy *= mask
                sum = np.sum(spgPolicy)
                if sum > 0:
                    spgPolicy /= np.sum(spgPolicy)
                else:
                    spgPolicy = mask
                node = node.expand(spgPolicy, node.board.turn == chess.WHITE)  # type: ignore
                node.backpropogate(spgValue)

        # TODO: Can we change those valuesn in SPG if we don't erase node and it should exist further?
        # ? Check this out
        actions = []
        for spg in spGames:
            actionProbs = np.zeros(ChessGame.actionSize)
            mask = np.zeros(ChessGame.actionSize)
            for child in spg.root.children:
                mask[child.actionTaken] = 1
                if child.visitCount != 0:
                    actionProbs[child.actionTaken] = child.visitCount

            actionProbs = actionProbs ** (1 / self.args["temperature"])

            actionsProbsSum = np.sum(actionProbs)
            if actionsProbsSum > 0:
                actionProbs /= actionsProbsSum
            else:
                actionProbs = mask
            spgVal = spg.root.valueSum / spg.root.visitCount
            spgVal = spgVal if spg.root.board.turn else -spgVal
            actions.append((spg.root.state, actionProbs, spgVal, mask))
        # self.drawer.update(spg.root)
        return actions, spGames
        # backprop

    # return visit counts
