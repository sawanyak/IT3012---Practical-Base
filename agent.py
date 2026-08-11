import random
# agent.py
class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        # If standing directly on food, or just wander / move towards coordinates
        # Simple heuristic or fallback random sweep
        return random.choice(self.actions_pool)


class SimpleReflexAgent:
    """Reacts to the current percept only, via fixed condition-action rules."""

    _LEFT_TURN = {'Up': 'Left', 'Left': 'Down', 'Down': 'Right', 'Right': 'Up'}

    def sense_and_act(self, percept: dict) -> str:
        facing = percept.get('facing', 'Right')

        if percept.get('food_here'):
            return facing                      # "suck" (food auto-collects on arrival)
        elif percept.get('wall_ahead'):
            return self._LEFT_TURN[facing]      # "turn_left"
        else:
            return facing                       # "move_forward"


class ModelBasedAgent:
    """A model-based reflex agent. It keeps an internal belief about which
    cell it occupies and remembers every cell it has already visited, so it
    can notice when a fixed reflex rule (e.g. "always turn left") would send
    it back into a loop, and break out via an alternate direction."""

    _LEFT_TURN = {'Up': 'Left', 'Left': 'Down', 'Down': 'Right', 'Right': 'Up'}
    _RIGHT_TURN = {'Left': 'Up', 'Down': 'Left', 'Right': 'Down', 'Up': 'Right'}
    _DELTA = {'Up': (0, 1), 'Down': (0, -1), 'Left': (-1, 0), 'Right': (1, 0)}

    def __init__(self):
        self.position = (0, 0)          # agent's belief about its own coordinates
        self.visited_cells = {(0, 0)}   # memory of every cell believed to have been occupied
        self.last_action = None
        self.last_percept = None

    def sense_and_act(self, percept: dict) -> str:
        facing = percept.get('facing', 'Right')

        # --- Transition & Sensor Model: fold the outcome of the last action into our belief state ---
        if self.last_action in self._DELTA and self.last_percept and not self.last_percept.get('wall_ahead'):
            dx, dy = self._DELTA[self.last_action]
            self.position = (self.position[0] + dx, self.position[1] + dy)
        self.visited_cells.add(self.position)

        # --- Condition-action rules, now consulting memory to avoid loops ---
        if percept.get('food_here'):
            action = facing
        elif percept.get('wall_ahead'):
            left_dir, right_dir = self._LEFT_TURN[facing], self._RIGHT_TURN[facing]
            left_cell = tuple(p + d for p, d in zip(self.position, self._DELTA[left_dir]))
            right_cell = tuple(p + d for p, d in zip(self.position, self._DELTA[right_dir]))

            if left_cell in self.visited_cells and right_cell not in self.visited_cells:
                action = right_dir          # left already explored -> break the loop
            else:
                action = left_dir           # default reflex behaviour
        else:
            action = facing

        self.last_action = action
        self.last_percept = percept
        return action
