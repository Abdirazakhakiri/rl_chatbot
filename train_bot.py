import csv
from rl_agent import RLAgent

agent = agent = RLAgent(actions=["responded"])

# Load and train on the CSV file
with open("Regenerated_100_Chatbot_Q_As.csv", newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        question = row['Question']
        answer = row['Answer']

        # Basic simulation of feedback
        reward = 1 if "book" in answer.lower() or "call" in answer.lower() else -1

        # Train using Q-learning
        agent.update(state=question, action="responded", reward=reward, next_state=question)

# Save the trained Q-table
agent.save_q_table()

print("Training complete. Q-table saved.")