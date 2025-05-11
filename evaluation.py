from contextlib import contextmanager,redirect_stderr,redirect_stdout
import os
import sys

from referee.agent import AgentProxyPlayer
# agents = ["MinimaxAgent", "ABAgent", "ABAgentWithCache", "MTCSAgent", "GreedyAgent", "RandomAgent"]
agents = ["RandomAgent", "ABAgentWithCache"]

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
with open("output.txt", "w") as f:
    for red in agents:
        for blue in agents:
    
                f.write(f"Running {red} vs {blue}\n")
                # suppress_stdout_stderr()
                result:AgentProxyPlayer = evaluation_referee([red, blue], time=180, space=250, use_colour=True, use_unicode=False, verbosity=3)[0]
                
                # enablePrint()
                f.write(f"{result}\n")
                f.write(f"--------------------------------\n")