"""Root entrypoint for launching the CastledEngine Pygame GUI (desktop and pygbag WASM)."""

import asyncio
from gui.main import main

if __name__ == "__main__":
    asyncio.run(main())

