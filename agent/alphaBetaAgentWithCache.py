# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent
from typing import Any
import numpy as np
import random
import time
from .game.board import Board, BoardMutation, BoardState, CellState, \
    ILLEGAL_RED_DIRECTIONS, ILLEGAL_BLUE_DIRECTIONS
from referee.game import (PlayerColor, Coord, Direction, Action, MoveAction,
                          GrowAction, IllegalActionException)
from referee.game.constants import MAX_TURNS

class TranspositionTable:
    def __init__(self):
        # random nunmber from 0->2^64, with 3 types of pieces, 7x7 board
        self.table = [[[random.randint(0, 18446744073709551616) for _ in range(4)] for _ in range(8)] for _ in range(8)]
        self.hash_table: dict[int, float] = {}
    def indexOf(self, cell: CellState) -> int:
        if cell == CellState("LilyPad"):
            return 0
        elif cell == CellState(PlayerColor.RED):
            return 1
        elif cell == CellState(PlayerColor.BLUE):
            return 2
        return -1
    def computeHash(self, board:Board) -> int:
        hash = 0
        for i in range(8):
            for j in range(8):
                if (board[Coord(i, j)] != CellState(None)):
                    piece = self.indexOf(board[Coord(i, j)])
                    hash ^= self.table[i][j][piece]
        return hash
    def insert(self, board:Board, eval_score: float):
        hash = self.computeHash(board)
        self.hash_table[hash] = eval_score
class ABAgentWithCache:
    """
    This class is the "entry point" for your agent, providing an interface to
    respond to various Freckers game events.
    """

    def __init__(self, color: PlayerColor, **referee: dict):
        """
        This constructor method runs when the referee instantiates the agent.
        Any setup and/or precomputation should be done here.
        """
        
        match color:
            case PlayerColor.RED:
                self._color = PlayerColor.RED
            case PlayerColor.BLUE:
                print("Testing: I am playing as BLUE")
                self._color = PlayerColor.BLUE
        self.board = Board()
        self.transposition_table = TranspositionTable()

    @property
    def opponent_color(self):
        return PlayerColor.BLUE if self._color == PlayerColor.RED else PlayerColor.RED
    def action(self, **referee: dict) -> Action:
        """
        This method is called by the referee each time it is the agent's turn
        to take an action. It must always return an action object. 
        """

        best_action = None
        best_score = -float("inf")
        next_possible_moves = self.get_next_possible_configurations(self.board, self._color)
        for action in next_possible_moves:
            if action is MoveAction and len(action.directions > 1):
                print(action)
            self.board.apply_action(action)
            search_limit = ((2.5) / len(next_possible_moves))
            score = self.iterative_deepening(current_state=self.board, time_limit=search_limit)
            self.board.undo_action()
            if score > best_score:
                best_score = score
                best_action = action
        return best_action
        

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        This method is called by the referee after a player has taken their
        turn. You should use it to update the agent's internal game state. 
        """

        try:
            self.board.apply_action(action=action)

        except IllegalActionException:
            print("Illegal action")

        # There are two possible action types: MOVE and GROW. Below we check
        # which type of action was played and print out the details of the
        # action for demonstration purposes. You should replace this with your
        # own logic to update your agent's internal game state representation.
        match action:
            case MoveAction(coord, dirs):
                dirs_text = ", ".join([str(dir) for dir in dirs])
                print(f"Testing: {color} played MOVE action:")
                print(f"  Coord: {coord}")
                print(f"  Directions: {dirs_text}")
            case GrowAction():
                print(f"Testing: {color} played GROW action")
            case _:
                raise ValueError(f"Unknown action type: {action}")
            
    def iterative_deepening(self, current_state: Board, time_limit: float) -> float:
        depth = 1
        end_time = time.time() + time_limit
        score = 0
        cutoff_depth = 150 - current_state.turn_count
        while True:
            if time.time() > end_time:
                break
            depth+=1
            score = self.minimax(current_state=current_state,
                                curDepth=0,
                                maxTurn=False,
                                targetDepth=depth,
                                alpha=-float("inf"),
                                beta=float("inf"))
            if depth > cutoff_depth:
                break 
        # print(self.transposition_table.hash_table)
        return score
    def minimax(self, current_state: Board,
                curDepth: int,
                maxTurn: bool,
                targetDepth: int,
                alpha : float,
                beta : float) -> float:
        
        hash = self.transposition_table.computeHash(current_state)
        if hash in self.transposition_table.hash_table:
            return self.transposition_table.hash_table[hash]
        # If the node is the leaf (at terminal state) or the maximum lookahead depth
        if curDepth == targetDepth or current_state.has_game_ended():
            eval_score = self.eval(current_state)
            self.transposition_table.insert(current_state, eval_score)
            return eval_score

        if maxTurn:
            value = -float("inf")
            for action in self.get_next_possible_configurations(
                    init_board=current_state, player_turn=self._color):
                current_state.apply_action(action)
                score = self.minimax(current_state, curDepth+1, False, targetDepth, alpha, beta)
                current_state.undo_action()
                # Get maximum score
                value = max(value, score)
                alpha = max(alpha, value) 
                if beta <= alpha:
                    break
                
            return value

        value = float("inf")
        for action in self.get_next_possible_configurations(
                init_board=current_state, player_turn=self.opponent_color):
            current_state.apply_action(action)
            score = self.minimax(current_state, curDepth+1, True, targetDepth, alpha, beta)
            current_state.undo_action()
            value = min(value, score)
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value
            
    def eval(self, board: Board):
        return - self.get_score(board, self._color.opponent) + self.get_score(board, self._color)

    def _is_valid_move(self, color: PlayerColor, direction: Direction):
        if color == PlayerColor.RED:
            return direction in {Direction.Down, Direction.DownLeft, Direction.DownRight, Direction.Left,
                                 Direction.Right}
        return direction in {Direction.Up, Direction.UpLeft, Direction.UpRight, Direction.Left, Direction.Right}
     # self implementation served for agent

    def get_next_possible_configurations(self, init_board: Board,
                                         player_turn: PlayerColor) -> list[Action]:
        if init_board.turn_count >= MAX_TURNS:
            return None
        # Try grow action
        grow_action = GrowAction()
        board = Board(initial_state=init_board._state, initial_player=player_turn)
        board.apply_action(grow_action)
        # Check if grow done anything
        check_change = False
        for cell in board._state:
            if board[cell] != init_board[cell]:
                check_change = True
                break
        if check_change:
            possible_configs = [grow_action]
        else:
            possible_configs = []


        # Acquire all respective colored frogs
        position_of_frogs: set[Coord] = set(
            coord for coord, cell in init_board._state.items()
            if cell.state == player_turn
        )
        # Try to move all of the frogs
        for cell in position_of_frogs:
            board = Board(init_board._state, initial_player=player_turn)
            neighbors = self.get_neighbors_and_distance(
                player_turn, board, cell,  [], [])
            for neighbor in neighbors:
                possible_configs.append(MoveAction(cell, neighbor))
        # Try to find possible paths if each of them do move action

        return possible_configs if len(possible_configs) != 0 else None

    def get_neighbors_and_distance(
            self,
            player_color: PlayerColor,
            init_board: Board,
            current_node: Coord,
            current_direction: list[Direction],
            visited: list[Coord] = [],
            can_jump_to_lilypad=True) -> list[list[Direction]]:
        """Get the neighbors and their distance from the current_node
        Args:
        boards: Dictionary with Coord instance key
        current_node: Current position
        current_direction: Tracking the direction to get to the node.
        visited: Tracking visited node. But set it default an empty list
        can_jump_to_lilypad: internal variable.

        Returns:

        """
        neighbor = []
        for direction in Direction:
            if not self._is_valid_move(player_color, direction):
                continue

            try: new_coord = current_node + direction
            except ValueError: continue

            if new_coord in visited: continue
            visited.append(new_coord)
            

            if init_board[new_coord] == CellState("LilyPad") and can_jump_to_lilypad:
                new_directions = current_direction.copy()
                new_directions.append(direction)
                neighbor.append(new_directions)
                
            elif init_board[new_coord] == CellState(PlayerColor.BLUE) or init_board[new_coord] == CellState(PlayerColor.RED):
                
                try: new_coord = new_coord + direction
                except ValueError: continue

                if init_board[new_coord] == CellState("LilyPad"):    
                    new_directions = current_direction.copy()
                    new_directions.append(direction)
                    neighbor.append(new_directions)
                    neighbor += self.get_neighbors_and_distance(player_color, init_board, new_coord, new_directions, visited, False)

        return neighbor
    def get_score(self, board: Board, player_color: PlayerColor) -> int:
        position_of_frogs: set[Coord] = set(
            coord for coord, cell in board._state.items()
            if cell.state == player_color
        )
        count = 0
        for cell in position_of_frogs:
                from_cell = 7 if player_color == PlayerColor.RED else 0
                count -= abs(from_cell - cell.r)
        return count / len(position_of_frogs)