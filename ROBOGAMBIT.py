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
  - Columns   = A–F (left to right)
  - Rows      = 6-1 (top to bottom)(from white's perspective)

Move output format:  "<piece_id>:<source_cell>-><target_cell>"
  e.g.  "1:B3->B4"   (White Pawn moves from B3 to B4)
"""

import numpy as np
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
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
# Move generation  
# ---------------------------------------------------------------------------

def get_pawn_moves(board: np.ndarray, row: int, col: int, piece: int):
    direction = 1 if is_white(piece) else -1
    moves = []
    start_row = 1 if is_white(piece) else 4
    promotion_row = 5 if is_white(piece) else 0  # last rank

    new_row = row + direction
    new_col = col

    if in_bounds(new_row, new_col) and board[new_row][new_col] == EMPTY:
        # check if landing on promotion row
        if new_row == promotion_row:
            promo_pieces = [WHITE_QUEEN, WHITE_BISHOP, WHITE_KNIGHT] if is_white(piece) else [BLACK_QUEEN, BLACK_BISHOP, BLACK_KNIGHT]
            for promo in promo_pieces:
                moves.append((piece, row, col, new_row, new_col, promo))
        else:
            moves.append((piece, row, col, new_row, new_col, None))
            if row == start_row:
                new_row2 = row + 2 * direction
                if in_bounds(new_row2, new_col) and board[new_row2][new_col] == EMPTY:
                    moves.append((piece, row, col, new_row2, new_col, None))

    # diagonal captures
    for i in [-1, 1]:
        cap_row = row + direction
        cap_col = col + i
        if in_bounds(cap_row, cap_col) and board[cap_row][cap_col] != EMPTY and not same_side(piece, board[cap_row][cap_col]):
            if cap_row == promotion_row:
                promo_pieces = [WHITE_QUEEN, WHITE_BISHOP, WHITE_KNIGHT] if is_white(piece) else [BLACK_QUEEN, BLACK_BISHOP, BLACK_KNIGHT]
                for promo in promo_pieces:
                    moves.append((piece, row, col, cap_row, cap_col, promo))
            else:
                moves.append((piece, row, col, cap_row, cap_col, None))

    return moves

def get_knight_moves(board: np.ndarray, row: int, col: int, piece: int):
    moves = []
    # for a knight the possible position increment can be (+-2,+-1) and (+-1,+-2) so we can do this by making two for loop O(n^2)
    for i in [-2,2]:
        for j in [-1,1]:
            # there will be two set of distinct coordinates of legal move for each pair of i,j so we can name them New_cord and new_cord2
            new_row = row+i
            new_col = col+j

            new_col2 = col +i
            new_row2 = row + j
            # we write two condition for each pair of coords which would check the bound of the new coord and if the new_coord is empty or a piece of different colour is avaible or not
            if in_bounds(new_row,new_col):
                if not same_side(piece,board[new_row,new_col]):
                   moves.append((piece,row,col,new_row,new_col))
            if in_bounds(new_row2,new_col2):
                if not same_side(piece,board[new_row2,new_col2]):
                   moves.append((piece,row,col,new_row2,new_col2))
            
    return moves


def get_sliding_moves(board: np.ndarray, row: int, col: int, piece: int, directions):
    """Generic sliding piece (bishop / queen / rook directions)."""
    moves = []
    for dr,dc in directions:
        new_row = row
        new_col = col
        # we write a While True loop traversing thruogh all the possible coords untill we reach any enemy or go out of bound
        while True:
            new_row += dr
            new_col += dc
            if in_bounds(new_row,new_col):
                if same_side(piece,board[new_row][new_col]): # break if we come in contact we friendly piece
                    break
                moves.append((piece,row,col,new_row,new_col))
                if board[new_row][new_col] != EMPTY:

                    break
            else:
                break

    return moves


def get_bishop_moves(board: np.ndarray, row: int, col: int, piece: int):
    diagonals = [(-1,-1),(-1,1),(1,-1),(1,1)]
    return get_sliding_moves(board, row, col, piece, diagonals)


def get_queen_moves(board: np.ndarray, row: int, col: int, piece: int):
    all_dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    return get_sliding_moves(board, row, col, piece, all_dirs)


def get_king_moves(board: np.ndarray, row: int, col: int, piece: int):
    moves = []
    # as we know the king can move in all the direction if no other friendly piece is there 
    for i in [-1,0,1]:
        for j in [-1,0,1]:
            if j == 0 and i==0:
                continue
            new_row = row +i 
            new_col = col +j
            if in_bounds(new_row,new_col) and not same_side(piece,board[new_row,new_col]):
                moves.append((piece,row,col,new_row,new_col))
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
    """Return list of (piece_id, src_row, src_col, dst_row, dst_col) for all legal moves."""
    moves = []
    ali = WHITE_PIECES if playing_white==True else BLACK_PIECES
    # we will use the White pieces list and black piecs list to identity which are opponents
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            piece = board[i][j]
            if piece in ali:
                gen = MOVE_GENERATORS[piece]
                moves.extend(gen(board,i,j,piece))
    return moves

def is_check(board,is_white_playing:bool=True):
    # here we check which site our king is 
    king_side = WHITE_KING if is_white_playing else BLACK_KING
    found = False
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] == king_side:
                king_row, king_col = row, col
                found = True
                break
        if found:
            break

    # King not on board so we consider as a check
    if not found:
        return True

    enemy_moves = get_all_moves(board, not is_white_playing)
    for move in enemy_moves:
        if move[3] == king_row and move[4] == king_col:
            return True
    return False

def all_legal_moves(board,is_white_playing):
    legal = []
    moves = get_all_moves(board,is_white_playing)
    for move in moves:
        new_board = apply_move(board,move)
        if not is_check(new_board,is_white_playing):
            legal.append(move)
    return legal
# ---------------------------------------------------------------------------
# Board evaluation heuristic  (TODO: tune weights / add positional tables)
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Piece-Square Tables (PSTs) — 6×6, from White's perspective.
# Higher = better square for that piece type.
# Black PSTs are the vertical mirror of White's.
# ---------------------------------------------------------------------------

# Pawns: reward advancement toward promotion
_PAWN_PST = np.array([
    [0,  0,  0,  0,  0,  0],   # row 0 — back rank (shouldn't be here)
    [5,  5,  5,  5,  5,  5],   # row 1 — starting row  (slight bonus)
    [10, 10, 15, 15, 10, 10],  # row 2
    [20, 20, 25, 25, 20, 20],  # row 3 — good central advance
    [35, 35, 40, 40, 35, 35],  # row 4 — near promotion
    [50, 50, 50, 50, 50, 50],  # row 5 — promotion rank
], dtype=float)

# Knights: love the centre, hate edges
_KNIGHT_PST = np.array([
    [-20, -10,  0,  0, -10, -20],
    [-10,   5, 10, 10,   5, -10],
    [  0,  10, 20, 20,  10,   0],
    [  0,  10, 20, 20,  10,   0],
    [-10,   5, 10, 10,   5, -10],
    [-20, -10,  0,  0, -10, -20],
], dtype=float)

# Bishops: reward diagonals and openness
_BISHOP_PST = np.array([
    [-10,  0,  0,  0,  0, -10],
    [  0, 10, 10, 10, 10,   0],
    [  0, 10, 15, 15, 10,   0],
    [  0, 10, 15, 15, 10,   0],
    [  0, 10, 10, 10, 10,   0],
    [-10,  0,  0,  0,  0, -10],
], dtype=float)

# Queen: active centre, but not too early
_QUEEN_PST = np.array([
    [-10, -5,  0,  0, -5, -10],
    [ -5,  5,  5,  5,  5,  -5],
    [  0,  5, 10, 10,  5,   0],
    [  0,  5, 10, 10,  5,   0],
    [ -5,  5,  5,  5,  5,  -5],
    [-10, -5,  0,  0, -5, -10],
], dtype=float)

# King: hide in the corner, penalise centre exposure
_KING_PST = np.array([
    [ 20, 25,  5,  5, 25,  20],
    [ 15, 15, -5, -5, 15,  15],
    [-10,-15,-20,-20,-15, -10],
    [-20,-25,-30,-30,-25, -20],
    [-30,-35,-40,-40,-35, -30],
    [-40,-45,-50,-50,-45, -40],
], dtype=float)

WHITE_PST = {
    WHITE_PAWN:   _PAWN_PST,
    WHITE_KNIGHT: _KNIGHT_PST,
    WHITE_BISHOP: _BISHOP_PST,
    WHITE_QUEEN:  _QUEEN_PST,
    WHITE_KING:   _KING_PST,
}
# Black PSTs are a vertical flip of White's
BLACK_PST = {
    BLACK_PAWN:   np.flipud(_PAWN_PST),
    BLACK_KNIGHT: np.flipud(_KNIGHT_PST),
    BLACK_BISHOP: np.flipud(_BISHOP_PST),
    BLACK_QUEEN:  np.flipud(_QUEEN_PST),
    BLACK_KING:   np.flipud(_KING_PST),
}


def evaluate(board: np.ndarray) -> float:
    """
    Static board evaluation from White's perspective.
    Positive  → advantage for White.  Negative → advantage for Black.

    Components (all O(36) — no move generation, no recursion):
      1. Material balance    — signed piece values
      2. Piece-square tables — reward good squares per piece type
      3. Doubled pawns       — penalty per extra pawn in same column

    NOTE: Mobility is intentionally excluded here. Calling get_all_moves()
    inside evaluate() would make every minimax leaf O(moves²), causing
    exponential slowdown. The search itself already rewards mobility
    implicitly (more moves = more branches = better scores bubble up).
    """
    score = 0.0

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            piece = board[row][col]
            if piece == EMPTY:
                continue

            # 1. Material
            score += PIECE_VALUES.get(piece, 0)

            # 2. Piece-square table
            if piece in WHITE_PST:
                score += WHITE_PST[piece][row][col]
            elif piece in BLACK_PST:
                score -= BLACK_PST[piece][row][col]

    # 3. Doubled-pawn penalty
    DOUBLED_PAWN_PENALTY = 20
    for col in range(BOARD_SIZE):
        wp = sum(1 for r in range(BOARD_SIZE) if board[r][col] == WHITE_PAWN)
        bp = sum(1 for r in range(BOARD_SIZE) if board[r][col] == BLACK_PAWN)
        if wp > 1:
            score -= DOUBLED_PAWN_PENALTY * (wp - 1)
        if bp > 1:
            score += DOUBLED_PAWN_PENALTY * (bp - 1)

    return score

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

def apply_move(board: np.ndarray, move) -> np.ndarray:
    piece, src_row, src_col, dst_row, dst_col = move[0], move[1], move[2], move[3], move[4]
    promotion = move[5] if len(move) > 5 else None

    new_board = board.copy()
    new_board[src_row][src_col] = EMPTY
    # if promotion, place the new piece — otherwise place the pawn
    new_board[dst_row][dst_col] = promotion if promotion is not None else piece
    return new_board

def get_capture_moves(board: np.ndarray, playing_white: bool):
    """Return only moves that capture an enemy piece."""
    captures = []
    for move in get_all_moves(board, playing_white):
        dst_row, dst_col = move[3], move[4]
        if board[dst_row][dst_col] != EMPTY:
            captures.append(move)
    return captures

# ---------------------------------------------------------------------------
# Format move string
# ---------------------------------------------------------------------------
def format_move(piece: int, src_row: int, src_col: int, dst_row: int, dst_col: int, promotion=None) -> str:
    src_cell = idx_to_cell(src_row, src_col)
    dst_cell = idx_to_cell(dst_row, dst_col)
    base = f"{piece}:{src_cell}->{dst_cell}"
    if promotion is not None:
        return f"{base}={promotion}"   # e.g. 1:B5->B6=4
    return base

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def minimax(board, depth, alpha, beta, is_maximising):
    if depth == 0:
        return quiescence(board, alpha, beta, is_maximising)

    if is_maximising:
        best = float('-inf')
        legal_moves = all_legal_moves(board, True)
        if not legal_moves:
            return -99999 + (10 - depth) if is_check(board, True) else 0
        for move in legal_moves:
            score = minimax(apply_move(board, move), depth - 1, alpha, beta, False)
            best = max(best, score)
            alpha = max(alpha, score)
            if beta <= alpha:
                break
        return best

    else:
        best = float('inf')
        legal_moves = all_legal_moves(board, False)
        if not legal_moves:
            return 99999 - (10 - depth) if is_check(board, False) else 0
        for move in legal_moves:
            score = minimax(apply_move(board, move), depth - 1, alpha, beta, True)
            best = min(best, score)
            beta = min(beta, score)
            if beta <= alpha:
                break
        return best

def quiescence(board, alpha, beta, is_maximising, qdepth=0):
    """
    Extend search at leaf nodes by exploring captures only.
    Prevents horizon effect — engine won't miss captures right after depth 0.
    qdepth cap prevents infinite recursion in long capture chains.
    """
    if qdepth >= 3:
        return evaluate(board)

    stand_pat = evaluate(board)

    if is_maximising:                                                       # for whites move
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)

        for move in get_capture_moves(board, True):
            new_board = apply_move(board, move)
            score = quiescence(new_board, alpha, beta, False, qdepth + 1)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return alpha

    else:                                                                  # for blacks move
        if stand_pat <= alpha:
            return alpha
        beta = min(beta, stand_pat)

        for move in get_capture_moves(board, False):
            new_board = apply_move(board, move)
            score = quiescence(new_board, alpha, beta, True, qdepth + 1)
            beta = min(beta, score)
            if beta <= alpha:
                break
        return beta


def get_best_move(board: np.ndarray, playing_white: bool = True) -> Optional[str]:
    """
    Given the current board state, return the best move string.

    Parameters
    ----------
    board        : 6×6 NumPy array representing the current game state.
    playing_white: True if the engine is playing as White, False for Black.
   

    Returns
    -------
    Move string in the format '<piece_id>:<src_cell>-><dst_cell>', or
    None if no legal moves are available.
    """
    depth = 4  # search depth (increase for stronger play, at cost of speed)

    legal_moves = all_legal_moves(board, playing_white)
    if not legal_moves:
        return None  # no legal moves — checkmate or stalemate
        
    best_move = legal_moves[0]
    best_score = float('-inf') if playing_white else float('inf')

    for move in legal_moves:
        new_board = apply_move(board, move)
        # After our move the opponent plays, so flip is_maximising
        score = minimax(new_board, depth - 1, float('-inf'), float('inf'), not playing_white)
        if playing_white and score > best_score:
            best_score = score
            best_move = move
        elif not playing_white and score < best_score:
            best_score = score
            best_move = move

    return format_move(*best_move)


# ---------------------------------------------------------------------------
# Quick smoke-test  
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example: standard-ish starting position on a 6x6 board
    # White pieces on rows 4-5, Black pieces on rows 0-1
    initial_board = np.array([
        [ 2,  3,  4,  5,  3,  2],   # Row 1 (A1–F1) — White back rank
        [ 1,  1,  1,  1,  1,  1],   # Row 2         — White pawns
        [ 0,  0,  0,  0,  0,  0],   # Row 3
        [ 0,  0,  0,  0,  0,  0],   # Row 4
        [ 6,  6,  6,  6,  6,  6],   # Row 5         — Black pawns
        [ 7,  8,  9, 10,  8,  7],   # Row 6 (A6–F6) — Black back rank
    ], dtype=int)

    print("Board:\n", initial_board)
    move = get_best_move(initial_board, playing_white=True)
    print("Best move for White:", move)
