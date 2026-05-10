import random

from config import CHECKMATE, DEPTH, PIECE_SQUARE_TABLES, PIECESCORE, STALEMATE


class TranspositionTable:
    """Simple transposition table for memoization"""

    def __init__(self):
        self.table = {}

    def get(self, board_hash, depth):
        key = (board_hash, depth)
        return self.table.get(key)

    def store(self, board_hash, depth, score, move):
        key = (board_hash, depth)
        self.table[key] = (score, move)

    def clear(self):
        self.table.clear()


transposition_table = TranspositionTable()


def findRandomMove(validMoves):
    """Select a random move from the list of valid moves."""
    return validMoves[random.randint(0, len(validMoves) - 1)]


def findBestMove(gs, validMoves):
    global nextMove, counter
    nextMove = None
    random.shuffle(validMoves)  # Shuffle to add randomness in AI's choice
    counter = 0  # Reset the counter for move evaluations
    findNegaMaxMoveWithAlphaBeta(
        gs,
        validMoves,
        DEPTH,
        1 if gs.white_to_move else -1,
        -float("inf"),
        float("inf"),
    )
    print(f"Evaluated {counter} positions")
    return nextMove


def orderMoves(gs, moves):
    """Order moves for better alpha-beta pruning"""

    def moveValue(move):
        # Prioritize captures
        if gs.board[move.endRow][move.endCol] != ".":
            captured_piece = gs.board[move.endRow][move.endCol][1]
            moving_piece = move.pieceMoved[1]
            # MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
            return PIECESCORE[captured_piece] - PIECESCORE[moving_piece]

        # Prioritize center control
        center_bonus = 0
        if move.endRow in [3, 4] and move.endCol in [3, 4]:
            center_bonus = 10

        return center_bonus

    return sorted(moves, key=moveValue, reverse=True)


def findNegaMaxMoveWithAlphaBeta(gs, validMoves, depth, turnMultiplier, alpha, beta):
    """Enhanced NegaMax with alpha-beta pruning and optimizations"""
    global nextMove, counter
    counter += 1

    if not validMoves:
        if gs.inCheck():
            return -CHECKMATE
        return STALEMATE

    # Check transposition table
    board_hash = hash(gs.board.tobytes())
    tt_entry = transposition_table.get(board_hash, depth)
    if tt_entry and depth < DEPTH:  # Don't use TT for root
        return tt_entry[0]

    if depth == 0:
        return turnMultiplier * scoreBoard(gs)

    # Order moves for better pruning
    validMoves = orderMoves(gs, validMoves)

    maxScore = -float("inf")
    bestMove = None

    for move in validMoves:
        gs.makeMove(move)
        score = -findNegaMaxMoveWithAlphaBeta(
            gs, gs.getValidMoves(), depth - 1, -turnMultiplier, -beta, -alpha
        )
        gs.undoMove()

        if score > maxScore:
            maxScore = score
            bestMove = move
            if depth == DEPTH:
                nextMove = move

        # Alpha-beta pruning
        alpha = max(alpha, maxScore)
        if alpha >= beta:
            break  # Beta cutoff

    # Store in transposition table
    if bestMove:
        transposition_table.store(board_hash, depth, maxScore, bestMove)

    return maxScore


def scoreBoard(gs):
    score = 0
    safety_score = 0
    pawn_score = 0
    mobility_score = 0

    white_king_pos = None
    black_king_pos = None
    white_pawn_counts = [0] * 8
    black_pawn_counts = [0] * 8

    board_list = gs.board.tolist()

    for row in range(8):
        for col in range(8):
            piece = board_list[row][col]
            if piece == ".":
                continue

            color = piece[0]
            piece_type = piece[1]
            piece_value = PIECESCORE[piece_type]

            if color == "w":
                positional_value = PIECE_SQUARE_TABLES[piece_type][row][col]
                score += piece_value + positional_value

                if piece_type == "p":
                    white_pawn_counts[col] += 1
                elif piece_type == "K":
                    white_king_pos = (row, col)
                elif piece_type == "N":
                    center_distance = abs(row - 3.5) + abs(col - 3.5)
                    mobility_score += max(0, 7 - center_distance) * 2
            else:
                positional_value = PIECE_SQUARE_TABLES[piece_type][7 - row][col]
                score -= piece_value + positional_value

                if piece_type == "p":
                    black_pawn_counts[col] += 1
                elif piece_type == "K":
                    black_king_pos = (row, col)
                elif piece_type == "N":
                    center_distance = abs(row - 3.5) + abs(col - 3.5)
                    mobility_score -= max(0, 7 - center_distance) * 2

    # Penalize exposed kings
    if white_king_pos and white_king_pos[0] > 1:
        safety_score -= 30
    if black_king_pos and black_king_pos[0] < 6:
        safety_score += 30

    # Penalize doubled pawns
    for c in range(8):
        if white_pawn_counts[c] > 1:
            pawn_score -= 10 * (white_pawn_counts[c] - 1)
        if black_pawn_counts[c] > 1:
            pawn_score += 10 * (black_pawn_counts[c] - 1)

    return score + safety_score + pawn_score + mobility_score
