# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent

# from .program import Agent as Agent
submission = False
if not submission:
    from .minimaxAgent import MinimaxAgent
    from .mctsAgent import MCTSAgent as MCTSAgent
    from .alphaBetaAgent import ABAgent
    from .alphaBetaAgentWithCacheV2 import ABAgentWithCache as ABAgentWithCacheV2
    from .randomAgent import RandomAgent
    from .greedyAgent import GreedyAgent
else:
    from .alphaBetaAgentWithCacheV2 import ABAgentWithCache as Agent