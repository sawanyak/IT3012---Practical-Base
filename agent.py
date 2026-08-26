from collections import deque
import heapq
import random

# agent.py
class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        # If standing directly on food, or just wander / move towards coordinates
        pos = percept['agent_pos']
        # Simple heuristic or fallback random sweep
        return random.choice(self.actions_pool)

class SearchAgent:
    """A goal-based agent that plans a full path to food using uninformed search."""

    # Action name -> (dx, dy). Matches execute_action's coordinate convention.
    ACTIONS = {
        'Up':    (0, 1),
        'Down':  (0, -1),
        'Left':  (-1, 0),
        'Right': (1, 0),
    }

    def __init__(self):
        self.plan = []
        self.active_algo = 'BFS'

    def step_cost(self, state) -> int:
        """Cost of moving into `state`. Uniform for now; change this to make UCS interesting."""
        return 1

    def get_successors(self, state, walls, grid_size):
        """Expand a node: returns [(action, next_state, cost), ...] for all legal moves."""
        width, height = grid_size
        x, y = state
        successors = []
        for action, (dx, dy) in self.ACTIONS.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in walls:
                successors.append((action, (nx, ny), self.step_cost((nx, ny))))
        return successors

    def bfs_search(self, start, goal, walls, grid_size):
        """FIFO queue -> expands shallowest nodes first. Optimal for uniform step costs."""
        frontier = deque([(start, [])])
        reached = {start}                       # graph search: never re-enqueue a state

        while frontier:
            state, path = frontier.popleft()    # FIFO
            if state == goal:
                return path
            for action, next_state, _ in self.get_successors(state, walls, grid_size):
                if next_state not in reached:
                    reached.add(next_state)
                    frontier.append((next_state, path + [action]))
        return []

    def dfs_search(self, start, goal, walls, grid_size):
        """LIFO stack -> expands deepest nodes first. Complete here, but NOT optimal."""
        frontier = [(start, [])]
        reached = set()                         # marked on expansion, not on generation

        while frontier:
            state, path = frontier.pop()        # LIFO
            if state == goal:
                return path
            if state in reached:
                continue
            reached.add(state)
            for action, next_state, _ in self.get_successors(state, walls, grid_size):
                if next_state not in reached:
                    frontier.append((next_state, path + [action]))
        return []

    def ucs_search(self, start, goal, walls, grid_size):
        """Priority queue ordered by g(n) -> expands the cheapest path first."""
        counter = 0                             # tie-breaker keeps the heap FIFO-stable
        frontier = [(0, counter, start, [])]
        best_cost = {start: 0}                  # cheapest g(n) found so far per state

        while frontier:
            cost, _, state, path = heapq.heappop(frontier)
            if state == goal:
                return path
            if cost > best_cost.get(state, float('inf')):
                continue                        # a cheaper route to this state already won
            for action, next_state, move_cost in self.get_successors(state, walls, grid_size):
                new_cost = cost + move_cost
                if new_cost < best_cost.get(next_state, float('inf')):
                    best_cost[next_state] = new_cost
                    counter += 1
                    heapq.heappush(frontier, (new_cost, counter, next_state, path + [action]))
        return []

    def sense_and_act(self, percept: dict) -> str:
        """Plan when we have no plan; otherwise just execute the next step of it."""
        if not self.plan:
            start = tuple(percept['agent_pos'])
            walls = set(percept['walls'])       # set -> O(1) lookups inside the search
            grid_size = percept['grid_size']
            food = percept['all_food']

            if not food:
                return 'Up'                     # nothing left to chase

            # Goal selection: closest pellet by Manhattan distance
            goal = min(food, key=lambda f: abs(f[0] - start[0]) + abs(f[1] - start[1]))

            if self.active_algo == 'BFS':
                self.plan = self.bfs_search(start, goal, walls, grid_size)
            elif self.active_algo == 'DFS':
                self.plan = self.dfs_search(start, goal, walls, grid_size)
            elif self.active_algo == 'UCS':
                self.plan = self.ucs_search(start, goal, walls, grid_size)
            else:
                raise ValueError(f"Unknown algorithm: {self.active_algo}")

            if not self.plan:
                return 'Up'                     # goal unreachable

        return self.plan.pop(0)
