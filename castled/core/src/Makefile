CC = gcc
CFLAGS = -std=c99 -O2 -Wall -Wextra
SRCS = bitboard.c magics.c zobrist.c board.c movegen.c eval.c tt.c search.c engine.c

.PHONY: perft clean

perft: $(SRCS) test_perft.c
	$(CC) $(CFLAGS) $(SRCS) test_perft.c -o test_perft
	./test_perft

clean:
	rm -f test_perft *.o

