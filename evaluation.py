from contextlib import contextmanager,redirect_stderr,redirect_stdout
import os
import sys

from referee.agent import AgentProxyPlayer
import json

from referee.game.player import PlayerColor
agents = ["ABAgentWithCache", "GreedyAgent", "RandomAgent", "MCTSAgent", "MinimaxAgent"]
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
with open("win.json", "a") as win_file:
    with open("usage.json", "a") as usage_file:
        win_df = {"data": []}
        usage_df = {"data": []}
        for i in range(1):
            for red in agents:
                for blue in agents:
                    try: 
                        result:AgentProxyPlayer = evaluation_referee([red, blue], time=180, space=250, use_colour=True, use_unicode=False, verbosity=3)[0]
                    except:
                        print(f"Exception Occured between red: {red} and blue: {blue}")
                        json.dump(win_df, win_file)
                        json.dump(usage_df, usage_file)
                        raise Exception("Premature exited")
                    # In case of draw
                    if result is None:
                        continue
                    try: 
                        if result is None:
                            win_df["data"].append(
                                {
                                    "round": i,
                                    "red_agent": red,
                                    "blue_agent": blue,
                                    "result": "draw"
                                }
                            )
                            continue
                        winner_name = result._name
                        winner_color = "RED" if result.color == PlayerColor.RED else "BLUE"
                        memory_used = result._agent.status.space_peak
                        time_elapsed = result._agent.status.time_used
                        win_df["data"].append(
                            {
                                "round": i,
                                "red_agent": red,
                                "blue_agent": blue,
                                "winner_name": winner_name,
                                "winner_color": winner_color
                            }
                        )
                        data = {
                            "round": i,
                            "red_agent": red,
                            "blue_agent": blue,
                            "winner_name": winner_name,
                            "winner_color": winner_color
                        }
                        data[winner_name] = {
                            "memory_used": memory_used,
                            "time_elapsed": time_elapsed
                        }
                        usage_df["data"].append(data)
                    except Exception as e:
                        win_df["data"].append(
                            {
                                "round": i,
                                "red_agent": red,
                                "blue_agent": blue,
                                "is_terminated_early": True,
                                "reasons": "An error Occured" + e.__str__()
                            }
                        )
                        usage_df["data"].append({
                            "round": i,
                            "red_agent": red,
                            "blue_agent": blue,
                            "is_terminated_early": True,
                            "reasons": "An error Occured" + e.__str__()
                        })
                        continue
        json.dump(win_df, win_file)
        json.dump(usage_df, usage_file)