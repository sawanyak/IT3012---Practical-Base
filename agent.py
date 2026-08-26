from collections import deque
import heapq
import random
import math

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

    def manhattan_distance(self, pos, goal) -> int:
        """h(n) = |x1 - x2| + |y1 - y2|  -- cost of a 4-way (no diagonal) grid walk."""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def euclidean_distance(self, pos, goal) -> float:
        """h(n) = sqrt((x1 - x2)^2 + (y1 - y2)^2)  -- straight-line distance."""
        return math.sqrt((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2)


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

    def astar_search(self, start_pos, goal_pos, walls, grid_size, heuristic_type='manhattan'):
        """Priority queue ordered by f(n) = g(n) + h(n) -> guided by the heuristic."""
        heuristic = (self.euclidean_distance if heuristic_type == 'euclidean'
                     else self.manhattan_distance)

        counter = 0                             # tie-breaker keeps the heap stable
        h_start = heuristic(start_pos, goal_pos)
        frontier = [(0 + h_start, 0, counter, start_pos, [])]   # (f, g, tie, pos, path)
        reached_states = set()                  # states we have already expanded

        while frontier:
            f_cost, g_cost, _, current_pos, path_taken = heapq.heappop(frontier)
            if current_pos == goal_pos:
                return path_taken
            if current_pos in reached_states:
                continue                        # a cheaper route to this state already won
            reached_states.add(current_pos)

            for action, next_pos, move_cost in self.get_successors(current_pos, walls, grid_size):
                if next_pos in reached_states:
                    continue
                g_new = g_cost + move_cost
                h_new = heuristic(next_pos, goal_pos)
                f_new = g_new + h_new
                counter += 1
                heapq.heappush(frontier, (f_new, g_new, counter, next_pos, path_taken + [action]))
        return []

    

    def sense_and_act(self, percept: dict) -> str:
        """Plan when we have no plan; otherwise just execute the next step of it."""
        if not self.plan:
            start = tuple(percept['agent_pos'])
            walls = set(percept['walls'])       # set -> O(1) lookups inside the search
            grid_size = percept['grid_size']
            food = percept['all_food']

            if percept['remaining_food'] == 0:
                return 'Up'                     # nothing left to chase

            # Goal selection: closest pellet by Manhattan distance
            goal = min(food, key=lambda f: self.manhattan_distance(start, f))

            if self.active_algo == 'BFS':
                self.plan = self.bfs_search(start, goal, walls, grid_size)
            elif self.active_algo == 'DFS':
                self.plan = self.dfs_search(start, goal, walls, grid_size)
            elif self.active_algo == 'UCS':
                self.plan = self.ucs_search(start, goal, walls, grid_size)

            elif self.active_algo == 'AStar':
                self.plan = self.astar_search(start, goal, walls, grid_size,
                                              heuristic_type='manhattan')
                
            else:
                raise ValueError(f"Unknown algorithm: {self.active_algo}")

            if not self.plan:
                return 'Up'                     # goal unreachable

        return self.plan.pop(0)



if __name__ == '__main__':
    # Step 1.1 testing checkpoint
    agent = SearchAgent()
    start, goal = (0, 0), (3, 4)
    print("Manhattan:", agent.manhattan_distance(start, goal))   # expected 7
    print("Euclidean:", agent.euclidean_distance(start, goal))   # expected 5.0

