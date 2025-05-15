# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent
from dataclasses import dataclass
from enum import Enum
from typing import Any
import time
from functools import cmp_to_key

from .utils import BOARD_WEIGHT_RED, BOARD_WEIGHT_BLUE
from .game.board import Board, CellState
from referee.game import (PlayerColor, Coord, Direction, Action, MoveAction,
                          GrowAction, IllegalActionException)
from referee.game.constants import MAX_TURNS
import numpy as np


class Flag(Enum):
    EXACT = 1
    ALPHA = 0
    BETA = -1

@dataclass
class TranspositionTableEntry:
    depth: int
    evalutation_score: float
    best_move: Action
    flag: Flag
    acient: int = 0

class TranspositionTable:
    """_summary_
    A class which serves as API to cache board state using Zobrist Hashing
    """
    def __init__(self):
        # random nunmber from 0->2^64, with 3 types of pieces, 7x7 board
        self.table = [[[np.random.randint(0, 2**63) for _ in range(4)] for _ in range(8)] for _ in range(8)]
        self.hash_table: dict[int, TranspositionTableEntry] = {}

    def indexOf(self, cell: CellState) -> int:
        """Map the cell state to an integer

        Args:
            cell (CellState): The cell state

        Returns:
            int: The appropriate mapping of the state, with Lilypad as 0, Red frog as 1, and Blue frog as 2
        """
        if cell == CellState("LilyPad"):
            return 0
        elif cell == CellState(PlayerColor.RED):
            return 1
        elif cell == CellState(PlayerColor.BLUE):
            return 2
        return -1
    
    def computeHash(self, board:Board) -> int:
        """Hash the board state using Zobrist hashing

        Args:
            board (Board): The board state

        Returns:
            int: a (almost) unique ID to the board state
        """
        hash_key = 0
        for i in range(8):
            for j in range(8):
                if (board[Coord(i, j)] != CellState(None)):
                    piece = self.indexOf(board[Coord(i, j)])
                    hash_key ^= self.table[i][j][piece]
        return hash_key
    def insert(self, board:Board, depth,  eval_score: float, best_move: Action, flag: Flag):
        """Insert a transposition table entry to the table

        Args:
            board (Board): Board State
            depth (_type_): Remaining depth to explore
            eval_score (float): Evaluation score, the upperbound (Beta), or the lowerbound (Alpha)
            best_move (Action): the killer move that caused the beta cutoff
            flag (Flag): Signs the evaluation score as exact evaluation, the upperbound (Beta), or the lowerbound value (Alpha)
        """
        hash = self.computeHash(board)
        # Check if already existed
        if hash in self.hash_table:
            # Replace it if depth > entry.depth
            if depth > self.hash_table[hash].depth:
                self.hash_table[hash] = TranspositionTableEntry(depth, eval_score, best_move, flag, 0)
        else:
            self.hash_table[hash] = TranspositionTableEntry(depth, eval_score, best_move, flag, 0)

    def lookup_entry(self, board: Board,  alpha: float, beta: float, depth: int) -> TranspositionTableEntry | None:
        """Return the entry

        Args:
            board (Board): board state
            alpha (float): current alpha (lowerbound) value
            beta (float): current beta (upperbound) value
            depth (int): remaning depth to assessed

        Returns:
            TranspositionTableEntry | None: The entry if exists, none if not
        """
        hash_key = self.computeHash(board)
        # check if hash key existed
        if hash_key not in self.hash_table:
            return None
        
        entry = self.hash_table[hash_key]
        # Avoid search instability by limiting search result
        if depth != entry.depth:
            return None
        return entry
    def lookup_move(self, board: Board,  alpha: float, beta: float, depth: int) -> Action | None:
        """Return the best move (if it has)

        Args:
            board (Board): board state
            alpha (float): lowerbound value
            beta (float): upperbound value
            depth (int): remaining search depth

        Returns:
            Action | None: Action if exists, otherwise None
        """
        hash_key = self.computeHash(board)
        # check if hash key existed
        if hash_key not in self.hash_table:
            return None
        entry = self.hash_table[hash_key]
        if depth > entry.depth:
            return None
        if entry.best_move is None:
            return None
        # match the exact PV node score
        if entry.flag == Flag.EXACT:
            return entry.best_move
        # match alpha (fail-low node) score
        if entry.flag == Flag.ALPHA and entry.evalutation_score <= alpha:
            return entry.best_move
        # match beta (fail-high node) score
        if entry.flag == Flag.BETA and entry.evalutation_score >= beta:
            return entry.best_move
    def clear_table(self):
        self.hash_table.clear()

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
        self.max_node_counts = 0

    @property
    def opponent_color(self):
        return PlayerColor.BLUE if self._color == PlayerColor.RED else PlayerColor.RED
    def action(self, **referee: dict) -> Action:
        """
        This method is called by the referee each time it is the agent's turn
        to take an action. It must always return an action object. 
        """
        # Clear the cache table
        self.transposition_table.clear_table()

        # Setting up the best action and its corresponding score
        best_action = None
        best_score = -float("inf")
        # Get next legal moves
        next_possible_moves = self.get_next_possible_configurations(self.board, self._color)
        search_limit = ((referee["time_remaining"]/(150-self.board.turn_count) )/ len(next_possible_moves))
        for action in next_possible_moves:
            self.board.apply_action(action)
            # Search for next mvoe            
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
        
        
    def iterative_deepening(self, current_state: Board, time_limit: float) -> float:
        depth = 1
        end_time = time.time() + time_limit
        score = 0
        cutoff_depth = MAX_TURNS - current_state.turn_count
        while True:
            # Limit the depth search based on 150 turns limit
            if depth > cutoff_depth:
                break
            if time.time() > end_time:
                break
            score = self.minimax(current_state=current_state,
                                curDepth=0,
                                maxTurn=False,
                                targetDepth=depth,
                                alpha=-float("inf"),
                                beta=float("inf"))
            depth+=1
            
        return score
    def actions_compare(self, a: Action, b: Action):
        """Aid function to sort action based on the length which allows the frog to travel, number of leaps jumps

        Args:
            a (Action): Action
            b (Action): Action

        Returns:
            _type_: _description_
        """
        if isinstance(a, GrowAction) and isinstance(b, GrowAction):
            return 0
        if isinstance(a, GrowAction) and isinstance(b, MoveAction):
            return -1
        if isinstance(b, GrowAction) and isinstance(a, MoveAction):
            return 1
        # Sort by the length of the directions
        length_directions_a = len(a.directions)
        length_directions_b = len(b.directions)
        # Sort by the vertical distance of the direction
        length_a = sum(1 if direction in {Direction.Down, Direction.DownLeft, Direction.DownRight, Direction.Up, Direction.UpLeft, Direction.UpRight} else 0 for direction in a.directions)
        length_b = sum(1 if direction in {Direction.Down, Direction.DownLeft, Direction.DownRight, Direction.Up, Direction.UpLeft, Direction.UpRight} else 0 for direction in b.directions)
        if length_a == length_b:
            return length_directions_a - length_directions_b
        return length_a - length_b
    def minimax(self, current_state: Board,
                curDepth: int,
                maxTurn: bool,
                targetDepth: int,
                alpha : float,
                beta : float) -> float:
        
        
        lookup_result = self.transposition_table.lookup_entry(current_state, alpha, beta, targetDepth - curDepth)
        if lookup_result is not None:
            if lookup_result.flag == Flag.EXACT:
                return lookup_result.evalutation_score
            # Bound search window
            if lookup_result.flag == Flag.ALPHA:
                alpha = max(alpha, lookup_result.evalutation_score-1)
            else:
                beta = min(beta, lookup_result.evalutation_score+1)

        # If the node is the leaf (at terminal state) or the maximum lookahead depth
        if curDepth >= targetDepth or current_state.has_game_ended(): 
            eval_score = self.eval(current_state)
            self.transposition_table.insert(current_state, targetDepth - curDepth, eval_score, None, Flag.EXACT)
            return eval_score

        if maxTurn:
            value = -float("inf")
            list_of_actions = self.get_next_possible_configurations(init_board=current_state, player_turn=self._color)
            list_of_actions.sort(key=cmp_to_key(self.actions_compare), reverse=True)
            lookup_result_move = self.transposition_table.lookup_move(current_state, alpha, beta, targetDepth - curDepth)
            # Move the best action to the beginning of the list if exists
            if lookup_result_move is not None:
                for i in range(0, len(list_of_actions)):
                    if list_of_actions[i] == lookup_result_move:
                        list_of_actions[i], list_of_actions[0] = list_of_actions[0], list_of_actions[i]
                        break
            
            for action in list_of_actions:
                current_state.apply_action(action)
                score = self.minimax(current_state, curDepth+1, False, targetDepth, alpha, beta)
                current_state.undo_action()
                # Get maximum score
                value = max(value, score)
                alpha = max(alpha, value) 
                # Beta cutoff (fail high)
                if  beta <= value:
                    self.transposition_table.insert(current_state, targetDepth - curDepth, value, action, Flag.BETA)
                    break
                
            return value

        value = float("inf")
        list_of_actions = self.get_next_possible_configurations(init_board=current_state, player_turn=self.opponent_color)
        list_of_actions.sort(key=cmp_to_key(self.actions_compare), reverse=True)

        for action in list_of_actions:
            current_state.apply_action(action)
            score = self.minimax(current_state, curDepth+1, True, targetDepth, alpha, beta)
            current_state.undo_action()

            value = min(value, score)
            beta = min(beta, value)

            # Alpha cutoff (fail-low node)
            if value <= alpha:
                self.transposition_table.insert(current_state, targetDepth - curDepth, value, None, Flag.ALPHA)
                break
        return value
            
    def eval(self, board: Board):
        
        return self.get_score(board, self._color) - self.get_score(board, self.opponent_color)

    def _is_valid_move(self, color: PlayerColor, direction: Direction):
        if color == PlayerColor.RED:
            return direction in {Direction.Down, Direction.DownLeft, Direction.DownRight, Direction.Left,
                                 Direction.Right}
        return direction in {Direction.Up, Direction.UpLeft, Direction.UpRight, Direction.Left, Direction.Right}


    def get_next_possible_configurations(self, init_board: Board,
                                         player_turn: PlayerColor) -> list[Action]:
        # The game should end after 150 turns
        if init_board.turn_count >= MAX_TURNS:
            return None
        
        board = Board(initial_state=init_board._state, initial_player=player_turn)
        board.apply_action(GrowAction())
        # Check if grow done anything
        check_change = False
        for cell in board._state.keys():
            if board[cell] != init_board[cell]:
                check_change = True
                break
        if check_change:
            possible_configs = [GrowAction()]
        else:
            possible_configs = []
        del board
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
            # Try new coordinate
            try: new_coord = current_node + direction
            except ValueError: continue

            # Try the next one if visited
            if new_coord in visited: continue
            visited.append(new_coord)
            
            # Try to jump to the lilypad
            if init_board[new_coord] == CellState("LilyPad") and can_jump_to_lilypad:
                new_directions = current_direction.copy()
                new_directions.append(direction)
                neighbor.append(new_directions)
            # Try to see if there's any free lilypad to perform a leap jump
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
        """Calculate the sum of weights based on all same-colour frogs positions

        Args:
            board (Board): Board state to evaluate
            player_color (PlayerColor): The frog colour to assess

        Returns:
            int: A sum of all frog's position weights
        """
        position_of_frogs: set[Coord] = set(
            coord for coord, cell in board._state.items()
            if cell.state == player_color
        )
        # Check if at the win position
        has_won = True
        dest_r = 7 if player_color == PlayerColor.RED else 0
        for cell in position_of_frogs:
            if cell.r != (dest_r):
                has_won = False
                break
        if has_won:
            return 2**32
        
        score = 0
        weight_score_matrix = BOARD_WEIGHT_RED if player_color == PlayerColor.RED else BOARD_WEIGHT_BLUE
        for cell in position_of_frogs:
            score += weight_score_matrix[cell.r][cell.c]
        return score