# UCI Protocol & Options

Start CastledEngine in UCI mode via:

```bash
castled uci
```

## Supported UCI Commands
- `uci` - Prints engine identity, supported options, and `uciok`.
- `isready` - Synchronizes engine state and replies `readyok`.
- `ucinewgame` - Resets board state and clears the transposition table.
- `setoption name <Name> value <Value>` - Updates an engine configuration option.
- `position [startpos | fen <fen>] [moves <m1> <m2> ...]` - Sets up the board state.
- `go [wtime <ms>] [btime <ms>] [winc <ms>] [binc <ms>] [movestogo <n>] [depth <d>] [nodes <n>] [movetime <ms>] [infinite]` - Starts background search and outputs `info` lines followed by `bestmove`.
- `stop` - Halts an ongoing search and emits `bestmove` immediately.
- `quit` - Terminates the UCI session cleanly.

## Configurable UCI Options

| Option Name | Type | Default | Range | Description |
|:---|:---:|:---:|:---:|:---|
| `Hash` | `spin` | `16` | `1..1024` | Transposition table size in MB |
| `Threads` | `spin` | `1` | `1..64` | Search thread count |
| `MoveOverhead` | `spin` | `10` | `0..5000` | Network/GUI latency buffer in ms |
| `UseNNUE` | `check` | `false` | `true/false` | Toggle C incremental NNUE vs classical eval |
| `EvalFile` | `string` | `<empty>` | Path | Path to custom `.nnue` binary weights |
| `BookFile` | `string` | `<empty>` | Path | Path to Polyglot `.bin` opening book |

