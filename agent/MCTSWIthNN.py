from dataclasses import dataclass
from .game.board import ILLEGAL_BLUE_DIRECTIONS, Board, CellState
from .program import Agent
from referee.game import PlayerColor, Action
import math

from referee.game.actions import GrowAction, MoveAction
from referee.game.board import ILLEGAL_RED_DIRECTIONS
from referee.game.constants import MAX_TURNS
from referee.game.coord import Coord, Direction
import random
import time
from collections import defaultdict
from referee.game.exceptions import IllegalActionException

class MCTSNode:
    def __init__(self, action: Action, board: Board, action_index: int):
        self.game = board
        self.action = action
        self.children = []
        self.parent = None
        self.visits = 0
        self._unexplored_actions = None
        self.result = defaultdict(int)
        self.result[1] = 0
        self.result[-1] = 0
        self.result[0] = 0
        self.action_index = action_index
        # the value of the node according to nn
        self.nn_v = 0
        
        # the next probabilities
        self.nn_p = None     
    
    def ubc_score(self, c: float)->float:
        if self.visits == 0 or self.parent is None:
            return float('inf')
        return (self.value / self.visits)+ c * math.sqrt((2*math.log(self.parent.visits)) / self.visits)
    @property
    def value(self):
        return self.result[1] + self.result[0] - self.result[-1]
    @property
    def unexplored_actions(self):
        if self._unexplored_actions is None:
            self._unexplored_actions = self.board.get_next_possible_configurations()
        return self._unexplored_actions
    
    def best_child(self, c: float = 0.01) -> 'MCTSNode':
        choices = [child for child in self.children if child.visits > 0]
        return max(choices, key=lambda child: child.ubc_score(c))
    def is_fully_expanded(self):
        return len(self.unexplored_actions) == 0
    @property
    def is_leaf(self):
        return len(self.children) == 0
    def is_terminal(self, color: PlayerColor):
        return self.board.determine_winner(color) is not None or self.board.turn_count >= MAX_TURNS
    def backpropagate(self, result: int):
        self.result[result] += 1
        self.visits += 1
        if self.parent:
            self.parent.backpropagate(result)
    def render(self):
        def format_node(node, depth=0):
            indent = "  " * depth
            output = f"{indent}Node Info:\n"
            output += f"{indent}Action: {node.action}\n"
            output += f"{indent}Visits: {node.visits}\n" 
            output += f"{indent}Value: {node.value}\n"
            output += f"{indent}UCB Score: {node.ubc_score(2)}\n"
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
        current = root
        iteration = 0
        start_time = time.time()
        while time.time() - start_time <= 180/75:
            iteration += 1
            current = self.tree_policy(root)
            new_board = Board(initial_state=current.board._state, 
                              initial_player=current.board.turn_color, 
                              turn_count=current.board.turn_count)
            result = self.rollout(new_board)
            current.backpropagate(result)
        print(root.render())
        print(max(root.children, key=lambda child: child.value/child.visits if child.visits > 0 else -float('inf')).render())
        return max(root.children, key=lambda child: child.value/child.visits if child.visits > 0 else -float('inf')).action
    def rollout(self, state: Board):

        while state.winner_color is None and state.turn_count < MAX_TURNS:
            action = self.rollout_policy(state)
            state.apply_action(action)
        if state.turn_count >= MAX_TURNS:
            return 0 if state.winner_color == None else 1 if state.winner_color == self._color else -1
        return 1 if state.winner_color == self._color else -1

    def try_action(self, state: Board, action: Action):
        state.apply_action(action)
        result = self.eval(state)
        state.undo_action()
        return result
    
    def rollout_policy(self, state: Board) -> Action:
        return max(state.get_next_possible_configurations(), 
                   key=lambda action: self.try_action(state, action))
    
    def tree_policy(self, root: MCTSNode) -> MCTSNode:
        current = root
        while not current.is_terminal(self._color):
            if not current.is_fully_expanded():
                return self.expand(current)
            else: 
                current = current.best_child()
        return current

    def expand(self, node: MCTSNode):

        action = node.unexplored_actions.pop()
        # Next state
        new_board = Board(initial_state=node.board._state, initial_player=node.board.turn_color, turn_count=node.board.turn_count)
        new_board.apply_action(action)
        # Create child node
        child = MCTSNode(action=action, board=new_board)
        child.parent = node
        node.children.append(child)
        
        return child