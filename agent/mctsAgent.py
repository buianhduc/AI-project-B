from dataclasses import dataclass
from agent.game.board import ILLEGAL_BLUE_DIRECTIONS, Board, CellState
from agent.program import Agent
from referee.game import PlayerColor, Action
import math

from referee.game.actions import GrowAction, MoveAction
from referee.game.board import ILLEGAL_RED_DIRECTIONS
from referee.game.constants import MAX_TURNS
from referee.game.coord import Coord, Direction
import random
import time

from referee.game.exceptions import IllegalActionException

class MCTSNode:
    def __init__(self, action: Action, board: Board):
        self.board = board
        self.action = action
        self.children = []
        self.parent = None
        self.visits = 0
        self.value = 0
        self._unexplored_actions = None
    
    def ubc_score(self, c: float)->float:
        if self.visits == 0 or self.parent is None:
            return float('inf')
        return self.value / self.visits + c * math.sqrt((2*math.log(self.parent.visits)) / self.visits)

    @property
    def unexplored_actions(self):
        if self._unexplored_actions is None:
            self._unexplored_actions = self.board.get_next_possible_configurations()
        return self._unexplored_actions
    
    def best_child(self, c: float = 1.41) -> 'MCTSNode':
        choices = [child for child in self.children if child.visits > 0]
        return max(choices, key=lambda child: child.ubc_score(c))
    def is_fully_expanded(self):
        return len(self.unexplored_actions) == 0
    @property
    def is_leaf(self):
        return len(self.children) == 0
    
    def backpropagate(self, result: int):
        self.value += result
        self.visits += 1
        if self.parent is not None:
            self.parent.backpropagate(result)
    def render(self):
        def format_node(node, depth=0):
            indent = "  " * depth
            output = f"{indent}Node Info:\n"
            output += f"{indent}Action: {node.action}\n"
            output += f"{indent}Visits: {node.visits}\n" 
            output += f"{indent}Value: {node.value}\n"
            output += f"{indent}UCB Score: {node.ubc_score}\n"
            output += f"{indent}Turn: {node.turn}\n"
            output += f"{indent}Children: {len(node.children)}\n"
            output += f"{indent}Board:\n{node.board.render()}\n"
            
            if node.children:
                output += f"{indent}Child Nodes:\n"
                for child in node.children:
                    output += format_node(child, depth + 1)
            
            return output
            
        return format_node(self)

class MCTSAgent(Agent):
    def __init__(self, color: PlayerColor,board:Board=None, **referee: dict):
        super().__init__(color)
        self.last_action_time = time.time()
        self.limit = 0
        self.board = Board() if board is None else board
        
    def action(self, **referee: dict) -> Action:  
        return self.search_best_action(self.board, referee['time_remaining'])
        
    
    def search_best_action(self, board: Board, remaining_time: float) -> Action:
        new_board = Board(initial_state=board._state, initial_player=self._color, turn_count=board.turn_count)
        root = MCTSNode(action=None, board=new_board)
        iteration = 0
        start_time = time.time()
        while time.time() - start_time < (remaining_time/(MAX_TURNS - self.board.turn_count)) + 10:
            current = self.tree_policy(root)
            new_board = Board(initial_state=current.board._state, 
                              initial_player=self._color, 
                              turn_count=current.board.turn_count)
            result = self.rollout(new_board)
            current.backpropagate(result)
            iteration += 1
        return root.best_child(c=0).action
    def rollout(self, state: Board):
        print("rollout turn count:", state.turn_count)
        while True:
            if state.determine_winner(self._color) is not None:
                print("rollout winner:", state.determine_winner(self._color))
                print(state.render())
                print("rollout turn:", state.turn_count)
                return state.determine_winner(self._color)
            
            action, _ = self.rollout_policy(state)
            state.apply_action(action)
            
    def rollout_policy(self, state: Board):
        return random.choice(state.get_next_possible_configurations())
    
    def tree_policy(self, root: MCTSNode) -> MCTSNode:
        current = root
        while not current.is_leaf or current.parent is None:
            if not current.is_fully_expanded():
                return self.expand(current)
            else: 
                current = current.best_child()
        return current

    def expand(self, node: MCTSNode):
        if len(node.unexplored_actions) > 0:
            action, board = node.unexplored_actions.pop()
            board.turn_count = node.board.turn_count + 1
            child = MCTSNode(action=action, board=board)
            child.parent = node
            node.children.append(child)
            return child