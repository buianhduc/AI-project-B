import math
from agent.game.board import Board, CellState
from referee.game.coord import Coord
from referee.game.player import PlayerColor
from .program import Agent


class GreedyAgent(Agent):
    def __init__(self, color, board = None, **referee):
        super().__init__(color, board, **referee)
    
    def action(self, **referee):
        best_action = None
        best_score = -float("inf")
        for action in self.board.get_next_possible_configurations():
            self.board.apply_action(action)
            score = self.eval(self.board)
            self.board.undo_action()
            if score > best_score:
                best_score = score
                best_action =action
        return best_action
            

    def eval(self, board):
        return self.get_score(board, self._color)
    
    def get_score(self, board: Board, player_color: PlayerColor) -> int:
        def distance(coord1: Coord, coord2: Coord):
            return math.floor(math.sqrt((coord1.c - coord2.c)**2 + (coord1.r - coord2.r)**2))
        position_of_frogs: set[Coord] = set(
            coord for coord, cell in board._state.items()
            if cell.state == player_color
        )
        count = 0
        for cell in position_of_frogs:
            if cell.r != (7 if player_color == PlayerColor.RED else 0):
                from_cell = (7 if player_color == PlayerColor.RED else 0)
                # check to the right
                left = cell.c
                right = cell.c
                while (left > 0 and board[Coord(from_cell, left)] == CellState(self._color)) and (right < 7 and board[Coord(from_cell, right)] == CellState(self._color)):
                    left -= 1
                    right += 1
                
                count -= min(abs(distance(cell, Coord(from_cell, left))), abs(distance(cell, Coord(from_cell,right))))
            # check to the left
            # count -= abs(from_cell - cell.r)
        return count