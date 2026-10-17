#include "board.h"
#include "zobrist.h"
#include "bitboard.h"
#include "magics.h"
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdio.h>

#define STARTING_FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

Board* board_new(void) {
    Board* b = (Board*)calloc(1, sizeof(Board));
    board_from_fen(b, STARTING_FEN);
    return b;
}

void board_free(Board* board) {
    free(board);
}

Board* board_copy(const Board* board) {
    Board* b = (Board*)malloc(sizeof(Board));
    memcpy(b, board, sizeof(Board));
    return b;
}

int board_piece_at(const Board* board, int sq) {
    return board->mailbox[sq];
}

int board_turn(const Board* board) {
    return board->turn;
}

int board_castling(const Board* board) {
    return board->castling;
}

int board_ep_square(const Board* board) {
    return board->ep_square;
}

int board_halfmove_clock(const Board* board) {
    return board->halfmove_clock;
}

int board_fullmove_number(const Board* board) {
    return board->fullmove_number;
}

uint64_t board_hash(const Board* board) {
    return board->hash;
}

void board_from_fen(Board* board, const char* fen) {
    memset(board->pieces, 0, sizeof(board->pieces));
    memset(board->occupied, 0, sizeof(board->occupied));
    for(int i=0; i<64; i++) board->mailbox[i] = PIECE_NONE;
    
    int sq = A8;
    while (*fen && *fen != ' ') {
        if (*fen == '/') { sq -= 16; fen++; continue; }
        if (isdigit(*fen)) { sq += (*fen - '0'); }
        else {
            int pc = PIECE_NONE;
            switch (*fen) {
                case 'P': pc = WHITE_PAWN; break; case 'N': pc = WHITE_KNIGHT; break;
                case 'B': pc = WHITE_BISHOP; break; case 'R': pc = WHITE_ROOK; break;
                case 'Q': pc = WHITE_QUEEN; break; case 'K': pc = WHITE_KING; break;
                case 'p': pc = BLACK_PAWN; break; case 'n': pc = BLACK_KNIGHT; break;
                case 'b': pc = BLACK_BISHOP; break; case 'r': pc = BLACK_ROOK; break;
                case 'q': pc = BLACK_QUEEN; break; case 'k': pc = BLACK_KING; break;
            }
            if (pc != PIECE_NONE) {
                board->pieces[pc] |= (1ULL << sq);
                board->mailbox[sq] = pc;
                board->occupied[pc < 6 ? WHITE : BLACK] |= (1ULL << sq);
                board->occupied[2] |= (1ULL << sq);
            }
            sq++;
        }
        fen++;
    }
    fen++;
    board->turn = (*fen == 'w') ? WHITE : BLACK;
    fen += 2;
    board->castling = 0;
    while (*fen && *fen != ' ') {
        if (*fen == 'K') board->castling |= WK;
        else if (*fen == 'Q') board->castling |= WQ;
        else if (*fen == 'k') board->castling |= BK;
        else if (*fen == 'q') board->castling |= BQ;
        fen++;
    }
    fen++;
    if (*fen != '-' && *fen != ' ') {
        int f = fen[0] - 'a';
        int r = fen[1] - '1';
        board->ep_square = r * 8 + f;
        fen += 2;
    } else {
        board->ep_square = SQ_NONE;
        fen++;
    }
    fen++;
    if (*fen) {
        sscanf(fen, "%d %d", &board->halfmove_clock, &board->fullmove_number);
    } else {
        board->halfmove_clock = 0; board->fullmove_number = 1;
    }
    board->hash = compute_hash(board);
    board->history_count = 0;
    board->hash_history_count = 0;
}

// Serialize current board state into a standard FEN string
void board_to_fen(const Board* board, char* buf, int bufsize) {
    static const char* symbols = "PNBRQKpnbrqk";
    int pos = 0;

    for (int r = 7; r >= 0; r--) {
        int empty = 0;
        for (int f = 0; f < 8; f++) {
            int pc = board->mailbox[r * 8 + f];
            if (pc == PIECE_NONE) {
                empty++;
            } else {
                if (empty > 0 && pos < bufsize - 1) buf[pos++] = '0' + empty;
                empty = 0;
                if (pos < bufsize - 1) buf[pos++] = symbols[pc];
            }
        }
        if (empty > 0 && pos < bufsize - 1) buf[pos++] = '0' + empty;
        if (r > 0 && pos < bufsize - 1) buf[pos++] = '/';
    }

    char castling[5];
    int cpos = 0;
    if (board->castling & WK) castling[cpos++] = 'K';
    if (board->castling & WQ) castling[cpos++] = 'Q';
    if (board->castling & BK) castling[cpos++] = 'k';
    if (board->castling & BQ) castling[cpos++] = 'q';
    if (cpos == 0) castling[cpos++] = '-';
    castling[cpos] = '\0';

    char ep[3] = "-";
    if (board->ep_square != SQ_NONE) {
        ep[0] = 'a' + (board->ep_square % 8);
        ep[1] = '1' + (board->ep_square / 8);
        ep[2] = '\0';
    }

    snprintf(buf + pos, bufsize - pos, " %c %s %s %d %d",
             (board->turn == WHITE) ? 'w' : 'b',
             castling, ep, board->halfmove_clock, board->fullmove_number);
}

static inline void put_piece(Board* board, int pc, int sq) {
    uint64_t bb = 1ULL << sq;
    board->pieces[pc] |= bb;
    board->mailbox[sq] = pc;
    board->occupied[pc < 6 ? WHITE : BLACK] |= bb;
    board->occupied[2] |= bb;
    board->hash ^= PIECE_KEYS[pc][sq];
}

static inline void remove_piece(Board* board, int sq) {
    int pc = board->mailbox[sq];
    if (pc == PIECE_NONE) return;
    uint64_t bb = 1ULL << sq;
    board->pieces[pc] &= ~bb;
    board->mailbox[sq] = PIECE_NONE;
    board->occupied[pc < 6 ? WHITE : BLACK] &= ~bb;
    board->occupied[2] &= ~bb;
    board->hash ^= PIECE_KEYS[pc][sq];
}

void board_make_move(Board* board, Move move) {
    int from = MOVE_FROM(move);
    int to = MOVE_TO(move);
    int flags = MOVE_FLAGS(move);
    int us = board->turn;
    
    State* st = &board->history[board->history_count++];
    st->move = move;
    st->castling = board->castling;
    st->ep_square = board->ep_square;
    st->halfmove_clock = board->halfmove_clock;
    st->hash = board->hash;
    st->captured_piece = board->mailbox[to];
    
    board->hash_history[board->hash_history_count++] = board->hash;
    
    int pc = board->mailbox[from];
    int captured = board->mailbox[to];
    
    if (board->ep_square != SQ_NONE) board->hash ^= EP_KEYS[board->ep_square % 8];
    board->ep_square = SQ_NONE;
    
    board->hash ^= CASTLING_KEYS[board->castling];
    board->castling &= CASTLING_RIGHTS_UPDATE[from] & CASTLING_RIGHTS_UPDATE[to];
    board->hash ^= CASTLING_KEYS[board->castling];
    
    board->halfmove_clock++;
    if (captured != PIECE_NONE || (pc % 6) == PAWN) {
        board->halfmove_clock = 0;
    }
    
    remove_piece(board, from);
    if (captured != PIECE_NONE) remove_piece(board, to);
    
    int promo_pc = PIECE_NONE;
    switch (flags) {
        case QUIET:
        case CAPTURE:
        case DOUBLE_PAWN:
            put_piece(board, pc, to);
            if (flags == DOUBLE_PAWN) {
                board->ep_square = (us == WHITE) ? to - 8 : to + 8;
                board->hash ^= EP_KEYS[board->ep_square % 8];
            }
            break;
        case EP_CAPTURE:
            put_piece(board, pc, to);
            int cap_sq = (us == WHITE) ? to - 8 : to + 8;
            st->captured_piece = board->mailbox[cap_sq];
            remove_piece(board, cap_sq);
            break;
        case KING_CASTLE:
            put_piece(board, pc, to);
            int r_from = (us == WHITE) ? H1 : H8;
            int r_to = (us == WHITE) ? F1 : F8;
            remove_piece(board, r_from);
            put_piece(board, (us == WHITE) ? WHITE_ROOK : BLACK_ROOK, r_to);
            break;
        case QUEEN_CASTLE:
            put_piece(board, pc, to);
            int qr_from = (us == WHITE) ? A1 : A8;
            int qr_to = (us == WHITE) ? D1 : D8;
            remove_piece(board, qr_from);
            put_piece(board, (us == WHITE) ? WHITE_ROOK : BLACK_ROOK, qr_to);
            break;
        default:
            if (flags == PROMO_N || flags == CAPTURE_PROMO_N) promo_pc = WHITE_KNIGHT;
            else if (flags == PROMO_B || flags == CAPTURE_PROMO_B) promo_pc = WHITE_BISHOP;
            else if (flags == PROMO_R || flags == CAPTURE_PROMO_R) promo_pc = WHITE_ROOK;
            else promo_pc = WHITE_QUEEN;
            if (us == BLACK) promo_pc += 6;
            put_piece(board, promo_pc, to);
            break;
    }
    
    board->turn ^= 1;
    board->hash ^= SIDE_KEY;
    if (us == BLACK) board->fullmove_number++;
}

void board_unmake_move(Board* board) {
    board->history_count--;
    board->hash_history_count--;
    State* st = &board->history[board->history_count];
    
    int us = board->turn ^ 1;
    board->turn = us;
    if (us == BLACK) board->fullmove_number--;
    
    int from = MOVE_FROM(st->move);
    int to = MOVE_TO(st->move);
    int flags = MOVE_FLAGS(st->move);
    
    int pc = board->mailbox[to];
    
    if (flags >= PROMO_N) {
        remove_piece(board, to);
        put_piece(board, (us == WHITE) ? WHITE_PAWN : BLACK_PAWN, from);
    } else {
        remove_piece(board, to);
        put_piece(board, pc, from);
    }
    
    if (flags == CAPTURE || (flags >= CAPTURE_PROMO_N && flags <= CAPTURE_PROMO_Q)) {
        put_piece(board, st->captured_piece, to);
    } else if (flags == EP_CAPTURE) {
        int cap_sq = (us == WHITE) ? to - 8 : to + 8;
        put_piece(board, st->captured_piece, cap_sq);
    } else if (flags == KING_CASTLE) {
        int r_from = (us == WHITE) ? H1 : H8;
        int r_to = (us == WHITE) ? F1 : F8;
        remove_piece(board, r_to);
        put_piece(board, (us == WHITE) ? WHITE_ROOK : BLACK_ROOK, r_from);
    } else if (flags == QUEEN_CASTLE) {
        int r_from = (us == WHITE) ? A1 : A8;
        int r_to = (us == WHITE) ? D1 : D8;
        remove_piece(board, r_to);
        put_piece(board, (us == WHITE) ? WHITE_ROOK : BLACK_ROOK, r_from);
    }

    board->castling = st->castling;
    board->ep_square = st->ep_square;
    board->halfmove_clock = st->halfmove_clock;
    board->hash = st->hash;
}

// Pass the turn without moving a piece for null-move pruning.
void board_make_null_move(Board* board) {
    State* st = &board->history[board->history_count++];
    st->move = 0;
    st->castling = board->castling;
    st->ep_square = board->ep_square;
    st->halfmove_clock = board->halfmove_clock;
    st->hash = board->hash;
    st->captured_piece = PIECE_NONE;

    board->hash_history[board->hash_history_count++] = board->hash;

    if (board->ep_square != SQ_NONE) {
        board->hash ^= EP_KEYS[board->ep_square % 8];
        board->ep_square = SQ_NONE;
    }

    board->halfmove_clock++;
    board->turn ^= 1;
    board->hash ^= SIDE_KEY;
}

// Restore board state after a null move.
void board_unmake_null_move(Board* board) {
    board->history_count--;
    board->hash_history_count--;
    State* st = &board->history[board->history_count];

    board->turn ^= 1;
    board->castling = st->castling;
    board->ep_square = st->ep_square;
    board->halfmove_clock = st->halfmove_clock;
    board->hash = st->hash;
}

uint64_t board_attackers_to(const Board* board, int sq, uint64_t occupied) {
    uint64_t attackers = 0;
    attackers |= PAWN_ATTACKS[WHITE][sq] & board->pieces[BLACK_PAWN];
    attackers |= PAWN_ATTACKS[BLACK][sq] & board->pieces[WHITE_PAWN];
    attackers |= KNIGHT_ATTACKS[sq] & (board->pieces[WHITE_KNIGHT] | board->pieces[BLACK_KNIGHT]);
    attackers |= KING_ATTACKS[sq] & (board->pieces[WHITE_KING] | board->pieces[BLACK_KING]);
    attackers |= bishop_attacks(sq, occupied) & (board->pieces[WHITE_BISHOP] | board->pieces[BLACK_BISHOP] | board->pieces[WHITE_QUEEN] | board->pieces[BLACK_QUEEN]);
    attackers |= rook_attacks(sq, occupied) & (board->pieces[WHITE_ROOK] | board->pieces[BLACK_ROOK] | board->pieces[WHITE_QUEEN] | board->pieces[BLACK_QUEEN]);
    return attackers;
}

bool board_is_attacked_by(const Board* board, int sq, int color) {
    return (board_attackers_to(board, sq, board->occupied[2]) & board->occupied[color]) != 0;
}

bool board_is_check(const Board* board) {
    int king_pc = (board->turn == WHITE) ? WHITE_KING : BLACK_KING;
    if (board->pieces[king_pc] == 0) return false;
    int king_sq = lsb(board->pieces[king_pc]);
    return board_is_attacked_by(board, king_sq, board->turn ^ 1);
}

bool board_is_repetition(const Board* board) {
    int count = 0;
    for (int i = 0; i < board->hash_history_count; i++) {
        if (board->hash_history[i] == board->hash) count++;
    }
    return count >= 2;
}

// True when neither side has enough material to force checkmate.
bool board_is_insufficient_material(const Board* board) {
    if (board->pieces[WHITE_PAWN] || board->pieces[BLACK_PAWN] ||
        board->pieces[WHITE_ROOK] || board->pieces[BLACK_ROOK] ||
        board->pieces[WHITE_QUEEN] || board->pieces[BLACK_QUEEN]) {
        return false;
    }

    int wn = popcount(board->pieces[WHITE_KNIGHT]);
    int bn = popcount(board->pieces[BLACK_KNIGHT]);
    int wb = popcount(board->pieces[WHITE_BISHOP]);
    int bb = popcount(board->pieces[BLACK_BISHOP]);
    int total_minors = wn + bn + wb + bb;

    if (total_minors <= 1) {
        return true;
    }

    if (total_minors == 2 && wn == 0 && bn == 0 && wb == 1 && bb == 1) {
        int w_sq = lsb(board->pieces[WHITE_BISHOP]);
        int b_sq = lsb(board->pieces[BLACK_BISHOP]);
        int w_color = ((w_sq / 8) + (w_sq % 8)) & 1;
        int b_color = ((b_sq / 8) + (b_sq % 8)) & 1;
        return w_color == b_color;
    }

    return false;
}

bool board_is_draw(const Board* board) {
    return board->halfmove_clock >= 100 || board_is_repetition(board) || board_is_insufficient_material(board);
}

