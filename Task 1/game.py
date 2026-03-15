"""
RoboGambit 2025-26 — Task 1: Autonomous Game Engine
Organised by Aries and Robotics Club, IIT Delhi

Board: 6x6 NumPy array
  - 0  : Empty cell
  - 1  : White Pawn
  - 2  : White Knight
  - 3  : White Bishop
  - 4  : White Queen
  - 5  : White King
  - 6  : Black Pawn
  - 7  : Black Knight
  - 8  : Black Bishop
  - 9  : Black Queen
  - 10 : Black King

Board coordinates:
  - Bpttom-left  = A1  (index [0][0])
  - Columns   = A-F (left to right)
  - Rows      = 6-1 (top to bottom)(from white's perspective)

Move output format:  "<piece_id>:<source_cell>-><target_cell>"
  e.g.  "1:B3->B4"   (White Pawn moves from B3 to B4)
"""

import numpy as np
from typing import Optional
import random
import time

# ---------------------------------------------------------------------------
# Constants  — DO NOT CHANGE
# ---------------------------------------------------------------------------

EMPTY = 0

# Piece IDs
WHITE_PAWN   = 1
WHITE_KNIGHT = 2
WHITE_BISHOP = 3
WHITE_QUEEN  = 4
WHITE_KING   = 5
BLACK_PAWN   = 6
BLACK_KNIGHT = 7
BLACK_BISHOP = 8
BLACK_QUEEN  = 9
BLACK_KING   = 10

WHITE_PIECES = {WHITE_PAWN, WHITE_KNIGHT, WHITE_BISHOP, WHITE_QUEEN, WHITE_KING}
BLACK_PIECES = {BLACK_PAWN, BLACK_KNIGHT, BLACK_BISHOP, BLACK_QUEEN, BLACK_KING}

BOARD_SIZE = 6

PIECE_VALUES = {
    WHITE_PAWN:   100,
    WHITE_KNIGHT: 300,
    WHITE_BISHOP: 320,
    WHITE_QUEEN:  900,
    WHITE_KING:  20000,
    BLACK_PAWN:  -100,
    BLACK_KNIGHT:-300,
    BLACK_BISHOP:-320,
    BLACK_QUEEN: -900,
    BLACK_KING: -20000,
}

# Column index → letter
COL_TO_FILE = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F'}
FILE_TO_COL = {v: k for k, v in COL_TO_FILE.items()}

# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def idx_to_cell(row: int, col: int) -> str:
    """Convert (row, col) zero-indexed to board notation e.g. (0,0) -> 'A1'."""
    return f"{COL_TO_FILE[col]}{row + 1}"

def cell_to_idx(cell: str):
    """Convert board notation e.g. 'A1' -> (row=0, col=0)."""
    col = FILE_TO_COL[cell[0].upper()]
    row = int(cell[1]) - 1
    return row, col

def in_bounds(row: int, col: int) -> bool:
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE

def is_white(piece: int) -> bool:
    return piece in WHITE_PIECES

def is_black(piece: int) -> bool:
    return piece in BLACK_PIECES

def same_side(p1: int, p2: int) -> bool:
    return (is_white(p1) and is_white(p2)) or (is_black(p1) and is_black(p2))

# ---------------------------------------------------------------------------
# Internal absolute piece values (for move ordering)
# ---------------------------------------------------------------------------

ABS_VALUES = {p: abs(v) for p, v in PIECE_VALUES.items()}

# ---------------------------------------------------------------------------
# Zobrist hashing  [OPT-1]
# ---------------------------------------------------------------------------

_rng = random.Random(0xDEADBEEF)

# ZOBRIST[piece_id][row][col]
ZOBRIST = [
    [[_rng.getrandbits(64) for _ in range(BOARD_SIZE)]
     for _ in range(BOARD_SIZE)]
    for _ in range(11)
]
ZOBRIST_SIDE = _rng.getrandbits(64)


def _board_to_zobrist(board: np.ndarray, is_white_turn: bool) -> int:
    """Compute full Zobrist hash from scratch (used at root only)."""
    h = 0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p = int(board[r, c])
            if p != EMPTY:
                h ^= ZOBRIST[p][r][c]
    if not is_white_turn:
        h ^= ZOBRIST_SIDE
    return h

# ---------------------------------------------------------------------------
# Transposition table  [OPT-1]
# ---------------------------------------------------------------------------

_TT_MAX   = 2_000_000
_TT_EXACT = 0
_TT_LOWER = 1
_TT_UPPER = 2

_tt: dict = {}
_killers: list = [[] for _ in range(64)]
_history: dict = {}
_nodes: int = 0


# Search deadline — set by get_best_move, checked inside _minimax
_deadline: float = float('inf')


def _clear_tables():
    global _tt, _killers, _history, _nodes, _deadline
    _tt = {}
    _killers = [[] for _ in range(64)]
    _history = {}
    _nodes = 0
    _deadline = float('inf')

# ---------------------------------------------------------------------------
# Move generation
# ---------------------------------------------------------------------------

def get_pawn_moves(board: np.ndarray, row: int, col: int, piece: int):
    """
    White Pawns move upward (increasing row index).
    Black Pawns move downward (decreasing row index).
    Captures are diagonal-forward.
    Includes double push from start row and promotion.
    """
    direction   = 1 if is_white(piece) else -1
    start_row   = 1 if is_white(piece) else 4
    promo_row   = 5 if is_white(piece) else 0
    promo_set   = [WHITE_QUEEN, WHITE_BISHOP, WHITE_KNIGHT] if is_white(piece) \
                  else [BLACK_QUEEN, BLACK_BISHOP, BLACK_KNIGHT]
    moves = []
    nr, nc = row + direction, col

    # Forward push
    if in_bounds(nr, nc) and board[nr, nc] == EMPTY:
        if nr == promo_row:
            for promo in promo_set:
                moves.append((piece, row, col, nr, nc, promo))
        else:
            moves.append((piece, row, col, nr, nc, None))
            # Double push from start row
            if row == start_row:
                nr2 = row + 2 * direction
                if in_bounds(nr2, nc) and board[nr2, nc] == EMPTY:
                    moves.append((piece, row, col, nr2, nc, None))

    # Diagonal captures
    for dc in (-1, 1):
        cr, cc = row + direction, col + dc
        if in_bounds(cr, cc) and board[cr, cc] != EMPTY and \
                not same_side(piece, board[cr, cc]):
            if cr == promo_row:
                for promo in promo_set:
                    moves.append((piece, row, col, cr, cc, promo))
            else:
                moves.append((piece, row, col, cr, cc, None))

    return moves


def get_knight_moves(board: np.ndarray, row: int, col: int, piece: int):
    moves = []
    for i in (-2, 2):
        for j in (-1, 1):
            for nr, nc in ((row + i, col + j), (row + j, col + i)):
                if in_bounds(nr, nc) and not same_side(piece, int(board[nr, nc])):
                    moves.append((piece, row, col, nr, nc))
    return moves


def get_sliding_moves(board: np.ndarray, row: int, col: int, piece: int, directions):
    """Generic sliding piece (bishop / queen / rook directions)."""
    moves = []
    for dr, dc in directions:
        nr, nc = row + dr, col + dc
        while in_bounds(nr, nc):
            target = int(board[nr, nc])
            if same_side(piece, target):
                break
            moves.append((piece, row, col, nr, nc))
            if target != EMPTY:
                break
            nr += dr
            nc += dc
    return moves


def get_bishop_moves(board: np.ndarray, row: int, col: int, piece: int):
    diagonals = [(-1,-1),(-1,1),(1,-1),(1,1)]
    return get_sliding_moves(board, row, col, piece, diagonals)


def get_queen_moves(board: np.ndarray, row: int, col: int, piece: int):
    all_dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    return get_sliding_moves(board, row, col, piece, all_dirs)


def get_king_moves(board: np.ndarray, row: int, col: int, piece: int):
    moves = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = row + dr, col + dc
            if in_bounds(nr, nc) and not same_side(piece, int(board[nr, nc])):
                moves.append((piece, row, col, nr, nc))
    return moves


MOVE_GENERATORS = {
    WHITE_PAWN:   get_pawn_moves,
    WHITE_KNIGHT: get_knight_moves,
    WHITE_BISHOP: get_bishop_moves,
    WHITE_QUEEN:  get_queen_moves,
    WHITE_KING:   get_king_moves,
    BLACK_PAWN:   get_pawn_moves,
    BLACK_KNIGHT: get_knight_moves,
    BLACK_BISHOP: get_bishop_moves,
    BLACK_QUEEN:  get_queen_moves,
    BLACK_KING:   get_king_moves,
}


def get_all_moves(board: np.ndarray, playing_white: bool):
    """Return list of pseudo-legal moves for the given side."""
    side  = WHITE_PIECES if playing_white else BLACK_PIECES
    moves = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p = int(board[r, c])
            if p in side:
                moves.extend(MOVE_GENERATORS[p](board, r, c, p))
    return moves

# ---------------------------------------------------------------------------
# Check detection
# ---------------------------------------------------------------------------

def _is_check(board: np.ndarray, is_white_playing: bool) -> bool:
    """Return True if the side's king is in check."""
    king = WHITE_KING if is_white_playing else BLACK_KING
    kr = kc = -1
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] == king:
                kr, kc = r, c
                break
        if kr != -1:
            break
    if kr == -1:
        return True
    for move in get_all_moves(board, not is_white_playing):
        if move[3] == kr and move[4] == kc:
            return True
    return False

# ---------------------------------------------------------------------------
# Make / Unmake  [OPT-2]
# ---------------------------------------------------------------------------

def _make(board: np.ndarray, move: tuple, zh: int):
    """Apply move in-place. Returns (captured, placed, new_zobrist)."""
    piece, sr, sc, dr, dc = move[0], move[1], move[2], move[3], move[4]
    promotion = move[5] if len(move) > 5 else None
    placed    = promotion if promotion is not None else piece
    captured  = int(board[dr, dc])
    h = zh
    h ^= ZOBRIST[piece][sr][sc]
    if captured != EMPTY:
        h ^= ZOBRIST[captured][dr][dc]
    h ^= ZOBRIST[placed][dr][dc]
    h ^= ZOBRIST_SIDE
    board[sr, sc] = EMPTY
    board[dr, dc] = placed
    return captured, placed, h


def _unmake(board: np.ndarray, move: tuple, captured: int, placed: int):
    """Restore board after _make."""
    board[move[1], move[2]] = move[0]
    board[move[3], move[4]] = captured

# ---------------------------------------------------------------------------
# Legal moves
# ---------------------------------------------------------------------------

def _legal_moves(board: np.ndarray, white: bool) -> list:
    """Filter pseudo-legal moves to those that leave king safe."""
    legal = []
    for move in get_all_moves(board, white):
        cap, placed, _ = _make(board, move, 0)
        if not _is_check(board, white):
            legal.append(move)
        _unmake(board, move, cap, placed)
    return legal

# ---------------------------------------------------------------------------
# apply_move — kept for backward compatibility
# ---------------------------------------------------------------------------

def apply_move(board: np.ndarray, piece, src_row, src_col, dst_row, dst_col) -> np.ndarray:
    """Non-destructive move application (template-compatible signature)."""
    new_board = board.copy()
    new_board[src_row][src_col] = EMPTY
    # Handle pawn promotion: auto-promote to queen on reaching the back rank
    if piece == WHITE_PAWN and dst_row == 5:
        new_board[dst_row][dst_col] = WHITE_QUEEN
    elif piece == BLACK_PAWN and dst_row == 0:
        new_board[dst_row][dst_col] = BLACK_QUEEN
    else:
        new_board[dst_row][dst_col] = piece
    return new_board

# ---------------------------------------------------------------------------
# Piece-Square Tables
# ---------------------------------------------------------------------------

_PAWN_PST = np.array([
    [0,  0,  0,  0,  0,  0],
    [5,  5,  5,  5,  5,  5],
    [10, 10, 15, 15, 10, 10],
    [20, 20, 25, 25, 20, 20],
    [35, 35, 40, 40, 35, 35],
    [50, 50, 50, 50, 50, 50],
], dtype=np.float32)

_KNIGHT_PST = np.array([
    [-20,-10,  0,  0,-10,-20],
    [-10,  5, 10, 10,  5,-10],
    [  0, 10, 20, 20, 10,  0],
    [  0, 10, 20, 20, 10,  0],
    [-10,  5, 10, 10,  5,-10],
    [-20,-10,  0,  0,-10,-20],
], dtype=np.float32)

_BISHOP_PST = np.array([
    [-10,  0,  0,  0,  0,-10],
    [  0, 10, 10, 10, 10,  0],
    [  0, 10, 15, 15, 10,  0],
    [  0, 10, 15, 15, 10,  0],
    [  0, 10, 10, 10, 10,  0],
    [-10,  0,  0,  0,  0,-10],
], dtype=np.float32)

_QUEEN_PST = np.array([
    [-10, -5,  0,  0, -5,-10],
    [ -5,  5,  5,  5,  5, -5],
    [  0,  5, 10, 10,  5,  0],
    [  0,  5, 10, 10,  5,  0],
    [ -5,  5,  5,  5,  5, -5],
    [-10, -5,  0,  0, -5,-10],
], dtype=np.float32)

_KING_PST = np.array([
    [ 20, 25,  5,  5, 25, 20],
    [ 15, 15, -5, -5, 15, 15],
    [-10,-15,-20,-20,-15,-10],
    [-20,-25,-30,-30,-25,-20],
    [-30,-35,-40,-40,-35,-30],
    [-40,-45,-50,-50,-45,-40],
], dtype=np.float32)

WHITE_PST = {
    WHITE_PAWN:   _PAWN_PST,
    WHITE_KNIGHT: _KNIGHT_PST,
    WHITE_BISHOP: _BISHOP_PST,
    WHITE_QUEEN:  _QUEEN_PST,
    WHITE_KING:   _KING_PST,
}
BLACK_PST = {
    BLACK_PAWN:   np.flipud(_PAWN_PST),
    BLACK_KNIGHT: np.flipud(_KNIGHT_PST),
    BLACK_BISHOP: np.flipud(_BISHOP_PST),
    BLACK_QUEEN:  np.flipud(_QUEEN_PST),
    BLACK_KING:   np.flipud(_KING_PST),
}

# Vectorised lookup arrays [OPT-8]
_MAT_ARR = np.zeros(11, dtype=np.float32)
for _p, _v in PIECE_VALUES.items():
    _MAT_ARR[_p] = _v

_PST_ARR = np.zeros((11, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
for _p, _pst in WHITE_PST.items():
    _PST_ARR[_p] = _pst
for _p, _pst in BLACK_PST.items():
    _PST_ARR[_p] = -_pst

# ---------------------------------------------------------------------------
# Board evaluation  [OPT-8]
# ---------------------------------------------------------------------------

def evaluate(board: np.ndarray) -> float:
    """
    Static board evaluation from White's perspective.
    Positive  -> advantage for White
    Negative  -> advantage for Black

    Components: material + PST (vectorised), doubled-pawn penalty,
    passed-pawn bonus, king-safety penalty.
    """
    rows = np.arange(BOARD_SIZE)
    score = float(
        _MAT_ARR[board].sum() +
        _PST_ARR[board, rows[:, None], rows[None, :]].sum()
    )

    DOUBLED_PENALTY = 20
    PASSED_BONUS    = 60
    KING_SAFETY_PEN = 40

    for col in range(BOARD_SIZE):
        col_data = board[:, col]
        wp = int(np.sum(col_data == WHITE_PAWN))
        bp = int(np.sum(col_data == BLACK_PAWN))

        if wp > 1:
            score -= DOUBLED_PENALTY * (wp - 1)
        if bp > 1:
            score += DOUBLED_PENALTY * (bp - 1)

        for row in range(BOARD_SIZE):
            if board[row, col] == WHITE_PAWN:
                if not np.any(board[row+1:, col] == BLACK_PAWN):
                    score += PASSED_BONUS
            elif board[row, col] == BLACK_PAWN:
                if not np.any(board[:row, col] == WHITE_PAWN):
                    score -= PASSED_BONUS

    for king, friendly, sign in (
            (WHITE_KING, WHITE_PIECES, -1),
            (BLACK_KING, BLACK_PIECES,  1)):
        pos = np.argwhere(board == king)
        if len(pos) == 0:
            continue
        kr, kc = int(pos[0, 0]), int(pos[0, 1])
        nb = sum(
            1 for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if not (dr == 0 and dc == 0)
            and in_bounds(kr+dr, kc+dc)
            and int(board[kr+dr, kc+dc]) in friendly
        )
        if nb < 2:
            score += sign * KING_SAFETY_PEN

    return score

# ---------------------------------------------------------------------------
# Capture-only move generation for quiescence  [OPT-7]
# ---------------------------------------------------------------------------

def _capture_moves(board: np.ndarray, white: bool) -> list:
    """MVV-LLA ordered pseudo-legal captures, filtered for king safety."""
    side = WHITE_PIECES if white else BLACK_PIECES
    raw  = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            p = int(board[r, c])
            if p not in side:
                continue
            for move in MOVE_GENERATORS[p](board, r, c, p):
                victim = int(board[move[3], move[4]])
                if victim != EMPTY:
                    raw.append((10 * ABS_VALUES.get(victim, 0)
                                 - ABS_VALUES.get(move[0], 0), move))
    raw.sort(key=lambda x: -x[0])
    legal = []
    for _, move in raw:
        cap, placed, _ = _make(board, move, 0)
        if not _is_check(board, white):
            legal.append(move)
        _unmake(board, move, cap, placed)
    return legal

# ---------------------------------------------------------------------------
# Move ordering  [OPT-4]
# ---------------------------------------------------------------------------

def _score_move(board: np.ndarray, move: tuple,
                depth: int, tt_best) -> int:
    if tt_best is not None and move[:5] == tt_best[:5]:
        return 1_000_000
    victim = int(board[move[3], move[4]])
    if victim != EMPTY:
        return 100_000 + 10 * ABS_VALUES.get(victim, 0) \
               - ABS_VALUES.get(move[0], 0)
    if len(move) > 5 and move[5] is not None:
        return 90_000
    killers = _killers[depth] if depth < 64 else []
    if len(killers) >= 1 and move == killers[0]:
        return 80_000
    if len(killers) >= 2 and move == killers[1]:
        return 79_000
    return min(_history.get((move[0], move[3], move[4]), 0), 70_000)


def _order(board, moves, depth, tt_best):
    return sorted(moves,
                  key=lambda m: _score_move(board, m, depth, tt_best),
                  reverse=True)

# ---------------------------------------------------------------------------
# Format move string
# ---------------------------------------------------------------------------

def format_move(piece: int, src_row: int, src_col: int,
                dst_row: int, dst_col: int) -> str:
    """Return move in required format: '<piece_id>:<source_cell>-><target_cell>'."""
    src_cell = idx_to_cell(src_row, src_col)
    dst_cell = idx_to_cell(dst_row, dst_col)
    return f"{piece}:{src_cell}->{dst_cell}"


def _format_full(move: tuple) -> str:
    """Internal: format move tuple including promotion if present."""
    piece, sr, sc, dr, dc = move[0], move[1], move[2], move[3], move[4]
    base = format_move(piece, sr, sc, dr, dc)
    if len(move) > 5 and move[5] is not None:
        return f"{base}={move[5]}"
    return base

# ---------------------------------------------------------------------------
# Quiescence search  [OPT-7]
# ---------------------------------------------------------------------------

def _quiescence(board: np.ndarray, alpha: float, beta: float,
                white: bool, zh: int, qdepth: int = 0) -> float:
    global _nodes
    _nodes += 1
    stand_pat = evaluate(board)
    DELTA = 900

    if white:
        if stand_pat >= beta:
            return beta
        if qdepth >= 5:
            return stand_pat
        if stand_pat + DELTA < alpha:
            return stand_pat
        alpha = max(alpha, stand_pat)
        for move in _capture_moves(board, True):
            cap, placed, nzh = _make(board, move, zh)
            score = _quiescence(board, alpha, beta, False, nzh, qdepth+1)
            _unmake(board, move, cap, placed)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return alpha
    else:
        if stand_pat <= alpha:
            return alpha
        if qdepth >= 5:
            return stand_pat
        if stand_pat - DELTA > beta:
            return stand_pat
        beta = min(beta, stand_pat)
        for move in _capture_moves(board, False):
            cap, placed, nzh = _make(board, move, zh)
            score = _quiescence(board, alpha, beta, True, nzh, qdepth+1)
            _unmake(board, move, cap, placed)
            beta = min(beta, score)
            if beta <= alpha:
                break
        return beta

# ---------------------------------------------------------------------------
# Minimax  [OPT-1,4,5,6]
# ---------------------------------------------------------------------------

def _minimax(board: np.ndarray, depth: int,
             alpha: float, beta: float,
             white: bool, zh: int,
             rep_hist: list,
             null_ok: bool = True) -> float:
    global _nodes
    _nodes += 1

    # Hard deadline check — abort cleanly
    if time.monotonic() >= _deadline:
        return 0

    # Repetition
    sk = (board.tobytes(), white)
    if rep_hist.count(sk) >= 2:
        return 0

    # TT lookup
    tt_entry = _tt.get(zh)
    tt_best  = None
    orig_alpha = alpha
    if tt_entry is not None:
        ts, td, tf, tt_best = tt_entry
        if td >= depth:
            if tf == _TT_EXACT:
                return ts
            elif tf == _TT_LOWER:
                alpha = max(alpha, ts)
            elif tf == _TT_UPPER:
                beta  = min(beta,  ts)
            if alpha >= beta:
                return ts

    if depth == 0:
        return _quiescence(board, alpha, beta, white, zh)

    moves = _legal_moves(board, white)
    if not moves:
        return (-(99000 + depth)) if _is_check(board, white) else 0

    moves = _order(board, moves, depth, tt_best)

    # Null move pruning  [OPT-5]
    has_pieces = any(
        int(board[r, c]) in (
            WHITE_PIECES - {WHITE_PAWN, WHITE_KING} if white
            else BLACK_PIECES - {BLACK_PAWN, BLACK_KING})
        for r in range(BOARD_SIZE) for c in range(BOARD_SIZE)
    )
    if null_ok and depth >= 3 and has_pieces and not _is_check(board, white):
        nzh = zh ^ ZOBRIST_SIDE
        # Pass the turn to opponent, search at reduced depth
        null_score = _minimax(board, depth-3, alpha, beta,
                              not white, nzh, rep_hist, null_ok=False)
        # If even giving opponent a free move doesn't hurt us, prune
        if white and null_score >= beta:
            return beta
        if not white and null_score <= alpha:
            return alpha

    rep_hist.append(sk)
    best_score = float('-inf') if white else float('inf')
    best_move  = None

    for idx, move in enumerate(moves):
        cap, placed, nzh = _make(board, move, zh)
        is_cap   = cap != EMPTY
        is_promo = len(move) > 5 and move[5] is not None
        reduce   = depth >= 3 and idx >= 4 and not is_cap and not is_promo

        if reduce:
            score = _minimax(board, depth-2, alpha, beta, not white, nzh, rep_hist)
            needs_full = (white and score > alpha) or (not white and score < beta)
            if needs_full:
                score = _minimax(board, depth-1, alpha, beta, not white, nzh, rep_hist)
        else:
            score = _minimax(board, depth-1, alpha, beta, not white, nzh, rep_hist)

        _unmake(board, move, cap, placed)

        if white:
            if score > best_score:
                best_score = score
                best_move  = move
            alpha = max(alpha, score)
        else:
            if score < best_score:
                best_score = score
                best_move  = move
            beta = min(beta, score)

        if beta <= alpha:
            if not is_cap and depth < 64:
                k = _killers[depth]
                if move not in k:
                    _killers[depth] = [move] + k[:1]
            hk = (move[0], move[3], move[4])
            _history[hk] = _history.get(hk, 0) + depth * depth
            break

    rep_hist.pop()

    # TT store
    if len(_tt) >= _TT_MAX:
        for k in list(_tt.keys())[:(_TT_MAX // 10)]:
            del _tt[k]
    if best_move is not None:
        if best_score <= orig_alpha:
            flag = _TT_UPPER
        elif best_score >= beta:
            flag = _TT_LOWER
        else:
            flag = _TT_EXACT
        _tt[zh] = (best_score, depth, flag, best_move)

    return best_score

# ---------------------------------------------------------------------------
# Main entry point  — MUST match template signature exactly
# ---------------------------------------------------------------------------

# Internal time budget per move (seconds). Aggressive: use full 2s.
_TIME_LIMIT = 2.0


def get_best_move(board: np.ndarray, playing_white: bool = True) -> Optional[str]:
    """
    Given the current board state, return the best move string.

    Parameters
    ----------
    board        : 6x6 NumPy array representing the current game state.
    playing_white: True if the engine is playing as White, False for Black.

    Returns
    -------
    Move string in the format '<piece_id>:<src_cell>-><dst_cell>', or
    None if no legal moves are available.

    Internal strategy: iterative deepening alpha-beta with Zobrist TT,
    in-place make/unmake, MVV-LLA move ordering, null-move pruning,
    LMR, and quiescence search. Burns the full _TIME_LIMIT budget.
    """
    global _nodes, _deadline

    _clear_tables()
    _nodes    = 0

    legal = _legal_moves(board, playing_white)
    if not legal:
        return None

    zh       = _board_to_zobrist(board, playing_white)
    sk       = (board.tobytes(), playing_white)
    rep_hist = [sk]

    best_move_tuple = legal[0]
    start      = time.monotonic()
    _deadline  = start + _TIME_LIMIT          # shared with _minimax
    soft_stop  = start + _TIME_LIMIT * 0.95  # don't start depth we can't finish

    for depth in range(1, 64):
        if time.monotonic() >= soft_stop:
            break

        alpha, beta   = float('-inf'), float('inf')
        iter_best     = legal[0]
        iter_score    = float('-inf') if playing_white else float('inf')
        completed     = True

        ordered = _order(board, legal, depth, None)
        for move in ordered:
            if time.monotonic() >= _deadline:
                completed = False
                break
            cap, placed, nzh = _make(board, move, zh)
            score = _minimax(board, depth-1, alpha, beta,
                             not playing_white, nzh, rep_hist)
            _unmake(board, move, cap, placed)

            if playing_white and score > iter_score:
                iter_score = score
                iter_best  = move
                alpha = max(alpha, score)
            elif not playing_white and score < iter_score:
                iter_score = score
                iter_best  = move
                beta = min(beta, score)

        if completed:
            best_move_tuple = iter_best
            legal = [iter_best] + [m for m in legal if m != iter_best]
        else:
            break

    return _format_full(best_move_tuple)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    initial_board = np.array([
        [ 2,  3,  4,  5,  3,  2],   # Row 1 (A1-F1) — White back rank
        [ 1,  1,  1,  1,  1,  1],   # Row 2         — White pawns
        [ 0,  0,  0,  0,  0,  0],   # Row 3
        [ 0,  0,  0,  0,  0,  0],   # Row 4
        [ 6,  6,  6,  6,  6,  6],   # Row 5         — Black pawns
        [ 7,  8,  9, 10,  8,  7],   # Row 6 (A6-F6) — Black back rank
    ], dtype=int)

    print("Board:\n", initial_board)
    move = get_best_move(initial_board, playing_white=True)
    print(move)