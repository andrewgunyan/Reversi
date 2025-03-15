import struct
import json
from collections import defaultdict

class WThorParser:
    def __init__(self, file_path, max_opening_moves=8):
        self.file_path = file_path
        self.max_opening_moves = max_opening_moves
        self.opening_book = defaultdict(lambda: defaultdict(int))  # {Board Hash: {Move: Count}}

    def parse_wthor(self):
        """Reads a .wtb file and extracts opening sequences."""
        with open(self.file_path, "rb") as f:
            f.read(16)  # Skip header (16 bytes)

            while True:
                game_data = f.read(68)  # Each game is 68 bytes
                if not game_data:
                    break
                
                moves = list(game_data[4:64])  # Extract moves (excluding metadata)
                self.process_game(moves)

    def process_game(self, moves):
        """Processes the first N moves of a game and updates the opening book."""
        board_state = ""  # Simple hash representation

        for i, move in enumerate(moves[:self.max_opening_moves]):
            if move == 0:  # No more moves
                break
            row, col = divmod(move - 1, 8)  # Convert move to (row, col)
            board_state += f"{row}{col}"
            self.opening_book[board_state][(row, col)] += 1  # Increment frequency

    def save_opening_book(self, filename="wthor_opening_book.json"):
        """Saves parsed openings with stringified keys at all levels."""
        with open(filename, "w") as f:
            json.dump(
                {str(k): {str(move): count for move, count in v.items()} for k, v in self.opening_book.items()},
                f,
                indent=2
            )
