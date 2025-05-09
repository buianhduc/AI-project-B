
from referee.game.actions import Action
from referee.game.player import PlayerColor
from agent.program import Agent
from agent.mctsAgent import MCTSAgent

class CombinedAgent(Agent):
    def __init__(self, color: PlayerColor, **referee: dict):
        super().__init__(color)
        self.mcts_agent = MCTSAgent(color, **referee)
        self.minimax_agent = Agent(color, **referee)

    def action(self, **referee: dict) -> Action:
        if self.board.turn_count < 10:
            return self.mcts_agent.action(**referee)
            
        else:
            return self.minimax_agent.action(**referee)
    def update(self, color: PlayerColor, action: Action, **referee: dict):
        self.board.apply_action(action)
        self.minimax_agent.update(color, action, **referee)
        self.mcts_agent.update(color, action, **referee)
