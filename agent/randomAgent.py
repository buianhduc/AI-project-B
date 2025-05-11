from .program import Agent
import random

class RandomAgent(Agent):
    def __init__(self, color, board = None, **referee):
        super().__init__(color, board, **referee)
    
    def action(self, **referee):
        return random.choice(self.board.get_next_possible_configurations())