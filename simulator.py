# simulator.py
import tkinter as tk
from visual_grid_game import GridGameGUI

def run_grid_hunt():
    root = tk.Tk()
    GridGameGUI(root, width=12, height=12, num_food=15, num_opponents=0)
    root.mainloop()

if __name__ == "__main__":
    run_grid_hunt()