import math
import chess.lib
from chess.lib.utils import encode, decode
from chess.lib.heuristics import evaluate
from chess.lib.core import makeMove

###########################################################################################
# Utility function: Determine all the legal moves available for the side.
# This is modified from chess.lib.core.legalMoves:
#  each move has a third element specifying whether the move ends in pawn promotion
def generateMoves(side, board, flags):
    for piece in board[side]:
        fro = piece[:2]
        for to in chess.lib.availableMoves(side, board, piece, flags):
            promote = chess.lib.getPromote(None, side, board, fro, to, single=True)
            yield [fro, to, promote]
            
###########################################################################################
# Example of a move-generating function:
# Randomly choose a move.
def random(side, board, flags, chooser):
    '''
    Return a random move, resulting board, and value of the resulting board.
    Return: (value, moveList, boardList)
      value (int or float): value of the board after making the chosen move
      moveList (list): list with one element, the chosen move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    moves = [ move for move in generateMoves(side, board, flags) ]
    if len(moves) > 0:
        move = chooser(moves)
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        value = evaluate(newboard)
        return (value, [ move ], { encode(*move): {} })
    else:
        return (evaluate(board), [], {})

###########################################################################################
# Stuff you need to write:
# Move-generating functions using minimax, alphabeta, and stochastic search.
def minimax(side, board, flags, depth):
    '''
    Return a minimax-optimal move sequence, tree of all boards evaluated, and value of best path.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''

    if depth == 0:
        return evaluate(board), [], {}

    possibleMoves = generateMoves(side, board, flags)

    # maximizing player
    if side == 0:

        maxVal = -9999999999
        movesTree = {}
        moveList = []

        for move in possibleMoves:

            fro, to, promote = move
            newSide, newBoard, newFlag = makeMove(side, board, fro, to, flags, promote)
            curVal, curMoveList, curMoveTree = minimax(newSide, newBoard, newFlag, depth - 1)
            movesTree[encode(*move)] = curMoveTree
            curMax = maxVal
            maxVal = max(curVal, maxVal)
            if maxVal > curMax:
                movesList = [ move ] + curMoveList
        
        return maxVal, movesList, movesTree

    # minimizing player
    else:

        minVal = 9999999999
        movesTree = {}
        moveList = []

        for move in possibleMoves:
            fro, to, promote = move
            newSide, newBoard, newFlag = makeMove(side, board, fro, to, flags, promote)
            curVal, curMoveList, curMoveTree = minimax(newSide, newBoard, newFlag, depth - 1)
            movesTree[encode(*move)] = curMoveTree
            curMin = minVal
            minVal = min(minVal, curVal)
            if minVal < curMin:
                moveList = [ move ] + curMoveList
        
        return minVal, moveList, movesTree


def alphabeta(side, board, flags, depth, alpha=-math.inf, beta=math.inf):
    '''
    Return minimax-optimal move sequence, and a tree that exhibits alphabeta pruning.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''

    if depth == 0:
        return evaluate(board), [], {}

    possibleMoves = generateMoves(side, board, flags)

    # maximizing player
    if side == 0:

        maxVal = -9999999999
        movesTree = {}
        moveList = []

        for move in possibleMoves:

            fro, to, promote = move
            newSide, newBoard, newFlag = makeMove(side, board, fro, to, flags, promote)
            curVal, curMoveList, curMoveTree = alphabeta(newSide, newBoard, newFlag, depth - 1, alpha, beta)
            movesTree[encode(*move)] = curMoveTree
            curMax = maxVal
            maxVal = max(curVal, maxVal)
            if maxVal > curMax:
                movesList = [ move ] + curMoveList


            alpha = max(curVal, alpha)
            if alpha >= beta:
                break
        
        return maxVal, movesList, movesTree

    # minimizing player
    else:

        minVal = 9999999999
        movesTree = {}
        moveList = []

        for move in possibleMoves:
            fro, to, promote = move
            newSide, newBoard, newFlag = makeMove(side, board, fro, to, flags, promote)
            curVal, curMoveList, curMoveTree = alphabeta(newSide, newBoard, newFlag, depth - 1, alpha, beta)
            movesTree[encode(*move)] = curMoveTree
            curMin = minVal
            minVal = min(minVal, curVal)
            if minVal < curMin:
                moveList = [ move ] + curMoveList


            beta = min(curVal, beta)
            if alpha >= beta:
                break
        
        return minVal, moveList, movesTree

def stochasticPath(side, board, flags, depth, breadth, chooser):

    if depth == 0:
        return evaluate(board), [], {}

    movesTree = {}
    # for some reason it needs a list
    possibleMoves = [ move for move in generateMoves(side, board, flags) ]
    thisMove = chooser(possibleMoves)
    fro, to, promote = thisMove
    newSide, newBoard, newFlag = makeMove(side, board, fro, to, flags, promote)
    newVal, newMoveList, newMoveTree = stochasticPath(newSide, newBoard, newFlag, depth-1, breadth, chooser)
    movesTree[encode(*thisMove)] = newMoveTree
    movesList = [ thisMove ] + newMoveList
    return newVal, movesList, movesTree

def stochastic(side, board, flags, depth, breadth, chooser):
    '''
    Choose the best move based on breadth randomly chosen paths per move, of length depth-1.
    Return: (value, moveList, moveTree)
      value (float): average board value of the paths for the best-scoring move
      moveLists (list): any sequence of moves, of length depth, starting with the best move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
      breadth: number of different paths 
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''

    possibleMoves = generateMoves(side, board, flags)
    bestAvgValue = 99999
    movesTree = {}
    for move in possibleMoves:
        totalVal = 0
        fro, to, promote = move
        firstSide, firstBoard, firstFlag = makeMove(side, board, fro, to, flags, promote)
        movesTree[(encode(*move))] = {}
        for i in range(breadth):
            curVal, moveList, moveTree = stochasticPath(firstSide, firstBoard, firstFlag, depth-1, breadth, chooser)
            for key in moveTree:
                movesTree[(encode(*move))][key] = moveTree[key]
            totalVal += curVal
        avgVal = float(totalVal) / float(breadth)
        if avgVal < bestAvgValue:
            bestAvgValue = avgVal
            bestMove = [ move ] + moveList

    return bestAvgValue, bestMove, movesTree
    
    
