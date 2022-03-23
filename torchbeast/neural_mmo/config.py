import nmmo
from .tasks import All


class DebugConfig(nmmo.config.Medium, nmmo.config.AllGameSystems):
    NPOP = 1
    NENT = 4

    @property
    def SPAWN(self):
        return self.SPAWN_CONCURRENT

    AGENT_LOADER = nmmo.config.TeamLoader
    AGENTS = NPOP * [nmmo.Agent]
    TASK = All