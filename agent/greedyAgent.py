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