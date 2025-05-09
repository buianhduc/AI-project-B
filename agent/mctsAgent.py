from agent.game.board import ILLEGAL_BLUE_DIRECTIONS, Board, CellState
from agent.program import Agent
from referee.game import PlayerColor, Action
import math

from referee.game.actions import GrowAction, MoveAction
from referee.game.board import ILLEGAL_RED_DIRECTIONS
from referee.game.coord import Coord, Direction
class MCTSNode:
    def __init__(self, 
                 turn: PlayerColor,
                 action: Action = None,
                 parent: 'MCTSNode' = None,
                 
                 ):
        self.action = action
        self.children = []
        self.parent = parent
        self.visits = 0
        self.number_of_win = 0
        self.total_playout = 0
        self.turn: PlayerColor = turn
    
    def is_leaf(self):
        return len(self.children) == 0
    
    def is_root(self):
        return self.parent is None
    @property
    def uct(self):
        if self.visits == 0:
            return float('inf')
        return self.value / self.visits + 2 * math.sqrt(math.log(self.parent.visits) / self.visits)
    
    def expand(self):
        self.children.append(MCTSNode(action=self.action, parent=self))
    def backpropagate(self, result: int, child: 'MCTSNode'):
        current_node = child
        while current_node != None:
            current_node.visits += 1
            current_node.number_of_win += result
            current_node = current_node.parent
        self.visits += 1
        self.number_of_win += result

    
class MCTSAgent(Agent):
    def __init__(self, color: PlayerColor, **referee: dict):
        super().__init__(color, **referee)
        self.board = Board()
        self.tree = MCTSNode()
    def action(self, **referee: dict) -> Action:
        pass
    
    def update(self, color: PlayerColor, action: Action, **referee: dict):
        super().update(color, action, **referee)
        
    def _is_valid_move(self, color: PlayerColor, direction: Direction):
        if color == PlayerColor.RED:
            return direction not in ILLEGAL_RED_DIRECTIONS
        return direction not in ILLEGAL_BLUE_DIRECTIONS
     # self implementation served for agent

    def get_next_possible_configurations(self, init_board: Board,
                                         player_turn: PlayerColor) -> list[tuple[Action, Board]]:
        """
            The function attempts to simulate the next possible configurations
            by trying to grow the board and then checking if it has changed, 
            and each frog can move in all directions if it has legal moves.
        """
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
            possible_configs = [(grow_action, board)]
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
            possible_configs += neighbors
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
            neighbors: list of tuples of action and board
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
    
    def mcts(self, board: Board):
        current_node = self.tree
        while (current_node.total_playout < 1000):
            if current_node.is_leaf():
                if current_node.total_playout == 0:
                    # rollout
                    pass
                else:
                    # for each available action from current node, add new state to tree
                    # current = first new node
                    # roll out
                    pass
                continue
            current_node = current_node.children.sort(key=lambda x: x.uct, reverse=True)[0]
    def rollout(self, board: Board):
        pass
    def simulate(self, board: Board):
        pass