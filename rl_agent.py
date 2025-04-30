
import json
import random

class RLAgent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.2):
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = {}
        self.load_q_table()

    def get_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate or state not in self.q_table:
            return random.choice(self.actions)
        return max(self.q_table[state], key=self.q_table[state].get)

    def update(self, state, action, reward, next_state):
        self.q_table.setdefault(state, {a: 0.0 for a in self.actions})
        self.q_table.setdefault(next_state, {a: 0.0 for a in self.actions})
        
        old_value = self.q_table[state][action]
        next_max = max(self.q_table[next_state].values())
        new_value = old_value + self.learning_rate * (reward + self.discount_factor * next_max - old_value)
        self.q_table[state][action] = new_value
        self.save_q_table()

    def save_q_table(self):
        with open("q_table.json", "w") as f:
            json.dump(self.q_table, f)

    def load_q_table(self):
        try:
            with open("q_table.json", "r") as f:
                self.q_table = json.load(f)
        except FileNotFoundError:
            self.q_table = {}
