"""lichess-bot launcher for ChessNet.

lichess-bot runs this file with the Python given as ``interpreter`` in the
configuration and passes ``engine_options`` as command-line flags, e.g.
``--checkpoint=... --uci --nodes=1000000``. Everything else is chessnet.play.
"""

from chessnet.play import main

if __name__ == "__main__":
    main()
