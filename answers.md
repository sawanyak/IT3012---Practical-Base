Answers for questions in the lab sheet 02

1. (Remember) According to Lecture 02, why is it impossible to program a mathematically perfect "Table-Driven Agent" for complex environments like Chess? What happens as the agent's lifetime increases?
* A table-driven agent works by storing a precomputed correct action for every possible situation it could encounter. But Chess alone has about 10⁴⁰ legal board states and 10¹²⁰ possible game sequences (the Shannon Number). No storage device in the universe is large enough to hold a table that size, and no one could ever write or compile it. As the agent's lifetime increases, the number of percept sequences it must account for grows combinatorially, making the table explode in size so table-driven agents become impossible for any complex, long-lived environment.


2. (Understand) Look at the code you wrote for your SimpleReflexAgent. Identify and explain the specific lines of code that represent the "Condition-Action Rules" discussed in the lecture.
* if percept.get('food_here'):
        return facing      # "suck" (food auto-collects on arrival)
    elif percept.get('wall_ahead'):
        return self._LEFT_TURN[facing]   # "turn_left"
    else:
        return facing         # "move_forward"

condition "food is in the current cell" maps to the action "collect/suck."
condition "wall ahead" maps to action "turn left."
the default rule: no obstruction means "move forward.
Each if/elif/else branch is a direct rule pairing one percept condition to one fixed action, with no memory of past percepts used.

3. (Analyze) Your SimpleReflexAgent likely got stuck in an infinite loop during Step 1.2. Based on the lecture, analyze exactly why this happened. How did the combination of "Partial Observability" and a lack of "Percept History" cause this failure?
* The agent only sees the immediate percept, it has no view of the maze beyond that and stores nothing about where it's already been. So when its fixed rule leads it back into a cell it already visited, it has no way to detect the repetition. The same condition always fires the same action, and it cycles forever instead of recognizing "I've been here before."

4. (Evaluate) In Step 1.3, you added an internal state to your ModelBasedAgent. Evaluate how your specific code handles the "Transition Model" (how the world evolves) and the "Sensor Model" (how the agent's actions affect the world).
 # --- Transition & Sensor Model: fold the outcome of the last action into our belief state ---
        if self.last_action in self._DELTA and self.last_percept and not self.last_percept.get('wall_ahead'):
            dx, dy = self._DELTA[self.last_action]
            self.position = (self.position[0] + dx, self.position[1] + dy)
        self.visited_cells.add(self.position)

* It applies the physics of movement (_DELTA[direction]) only if the last action was a move and no wall blocked it, updating self.position to match how the world actually evolved. It folds that into the sensor/history model by recording the new cell in visited_cells, so the agent accumulates a memory of state the current percept alone can't reveal. It then close the loop: before repeating a reflex turn, it checks whether the cell that turn leads to (via the transition model) is already visited_cells, and picks the unexplored direction instead using its belief state to override a rule that would otherwise repeat the infinite loop.
