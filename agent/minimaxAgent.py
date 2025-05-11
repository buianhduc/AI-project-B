# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent
from typing import Any

from referee.game.constants import MAX_TURNS

from .game.board import Board, BoardMutation, BoardState, CellState, \
    ILLEGAL_RED_DIRECTIONS, ILLEGAL_BLUE_DIRECTIONS
from referee.game import (PlayerColor, Coord, Direction, Action, MoveAction,
                          GrowAction, IllegalActionException)


class MinimaxAgent:
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

    @property
    def opponent_color(self):
        return PlayerColor.BLUE if self._color == PlayerColor.RED else PlayerColor.RED
    def action(self, **referee: dict) -> Action:
        """
        This method is called by the referee each time it is the agent's turn
        to take an action. It must always return an action object. 
        """
        
        copy = Board(self.board._state, initial_player=self._color)
        best_action = None
        best_score = -float("inf")
        for action in copy.get_next_possible_configurations():
            copy.apply_action(action)
            score  = self.minimax(current_state=copy,
                        curDepth=0,
                        maxTurn=False,
                        targetDepth=2)
            copy.undo_action()
            if best_score < score:
                best_action = action
                best_score = score
        # Reconstruct
        return best_action
        

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        This method is called by the referee after a player has taken their
        turn. You should use it to update the agent's internal game state. 
        """

        try:
            # self.board.set_turn_color(color)
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
    def minimax(self, current_state: Board,
                curDepth: int,
                maxTurn: bool,
                targetDepth: int) -> int:
        # If the node is the leaf (at terminal state) or the maximum lookahead depth
        if curDepth == targetDepth or current_state.has_game_ended():
            return self.eval(current_state)

        if maxTurn:
            value = -float("inf")

            for action in self.get_next_possible_configurations(
                    init_board=current_state, player_turn=self._color):
                
                current_state.apply_action(action)
                score = self.minimax(current_state, curDepth+1, False,
                                            targetDepth)
                current_state.undo_action()
                if value < score:
                    value = score

            return value

        value = float("inf")
        for action in self.get_next_possible_configurations(
                init_board=current_state, player_turn=self.opponent_color):

            current_state.apply_action(action)
            score = self.minimax(current_state, curDepth+1, True,
                                          targetDepth)
            current_state.undo_action()
            if value > score:
                value = score
        return value
            
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
            neighbors = init_board.get_neighbors_and_distance(
                player_turn, board, cell,  [], [])

            possible_configs += [MoveAction(cell, directions) for directions in neighbors]
        # Try to find possible paths if each of them do move action

        return possible_configs if len(possible_configs) != 0 else None

    def get_neighbors_and_distance(
            self,
            player_color: PlayerColor,
            init_board: Board,
            current_node: Coord,
            current_direction: list[Direction],
            visited: list[Coord] = [],
            can_jump_to_lilypad=True) -> list[tuple[Action, Board]]:
        """Get the neighbors and their distance from the current_node
        Args:
        boards: Dictionary with Coord instance key
        current_node: Current position
        current_direction: Tracking the direction to get to the node.
        visited: Tracking visited node. But set it default an empty list
        can_jump_to_lilypad: internal variable.

        Returns:

        """

        neighbors = []
        for direction in Direction:

            if self._is_valid_move(player_color, direction):

                # Try if it's a valid move, if not continue to the next directio
                try: new_coord = current_node + direction
                except ValueError: continue
                # If that new coord has already been visited
                if new_coord in visited:
                    continue
                visited.append(new_coord)
                board = Board(init_board._state, player_color)
                # If the next node is a lilypad, and was not over another frog in the previous move
                if (init_board[new_coord] == CellState("LilyPad") and
                        can_jump_to_lilypad):
                    new_dir = current_direction.copy()
                    new_dir.append(direction)

                    board.apply_action(MoveAction(current_node, *new_dir))
                    neighbors.append((MoveAction(current_node, *new_dir),board))
                # If the next node is a frog, check if it can be jumped over
                elif (init_board[new_coord] == CellState(PlayerColor.BLUE) or
                      init_board[new_coord] == CellState(PlayerColor.RED)):

                    try:
                        new_coord += direction
                        if init_board[new_coord] == CellState("LilyPad"):
                            new_dir = current_direction.copy()
                            new_dir.append(direction)
                            neighbors += self.get_neighbors_and_distance(
                                player_color, board, new_coord, new_dir,
                                visited, False)
                    except ValueError:
                        continue
        return neighbors
    
    def eval(self, board: Board):
        return self.get_score(board, self._color) - self.get_score(board, self.opponent_color)
    
    def get_score(self, board: Board, player_color: PlayerColor) -> int:
        position_of_frogs: set[Coord] = set(
            coord for coord, cell in board._state.items()
            if cell.state == player_color
        )
        count = 0
        for cell in position_of_frogs:
                from_cell = 7 if player_color == PlayerColor.RED else 0
                count -= abs(from_cell - cell.r)
        return count