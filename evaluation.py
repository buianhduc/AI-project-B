from contextlib import contextmanager,redirect_stderr,redirect_stdout
import os
import sys

from referee.agent import AgentProxyPlayer
import json

from referee.game.player import PlayerColor
agents = ["ABAgentWithCache", "ABAgent"]
from referee import evaluation_referee
from os import devnull

# Disable
save = []
null_fds = []
@contextmanager
def suppress_stdout_stderr():
        # open 2 fds
    null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
    # save the current file descriptors to a tuple
    save = os.dup(1), os.dup(2)
    # put /dev/null fds on 1 and 2
    os.dup2(null_fds[0], 1)
    os.dup2(null_fds[1], 2)
# Restore
def enablePrint():
        # restore file descriptors so I can print the results
    os.dup2(save[0], 1)
    os.dup2(save[1], 2)
    # close the temporary fds
    os.close(null_fds[0])
    os.close(null_fds[1])

with open("resultABs.json", "a") as win_file:
    win_df = []
    for i in range(20):
        red = "ABAgentWithCacheV2" if i % 2 == 0 else "ABAgent"
        blue = "ABAgentWithCacheV2" if i % 2 != 0 else "ABAgent"
        try: 
            result:dict = evaluation_referee([red,blue], time=180, space=250, use_colour=True, use_unicode=False, verbosity=3)[0]
        except:
            print(f"Exception Occured between red: {red} and blue: {blue}")
            json.dump(win_df, win_file)
            raise Exception("Premature exited")
        win_df.append(
            {
                "round": i,
                "red_agent": {
                    "name": result["player1"]._name,
                    "time_elapsed": result["player1"]._agent.status.time_used,
                    "memory_peak": result["player1"]._agent.status.space_peak,
                },
                "blue_agent": {
                    "name": result["player2"]._name,
                    "time_elapsed": result["player2"]._agent.status.time_used,
                    "memory_peak": result["player2"]._agent.status.space_peak,
                },
                "result": "Draw" if result["result"] is None else "Red" if result["result"] == PlayerColor.RED else "BLUE"

            }
        )
    json.dump(win_df, win_file)
                