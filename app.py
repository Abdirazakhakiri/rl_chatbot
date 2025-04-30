
from flask import Flask, request, jsonify
from rl_agent import RLAgent

app = Flask(__name__)

# Initialize the agent with possible responses
agent = RLAgent(actions=[
    "Want to book a free call?",
    "Would you like help with Facebook Ads?",
    "Can I help you set up your funnel?",
    "Ready to book a strategy call?",
    "Need help attracting leads?"
])

# Load the trained Q-table
agent.load_q_table()

REWARD_SUCCESS = 1
REWARD_FAILURE = -1

@app.route('/webhook', methods=['POST'])  # fixed "methds" typo here
def webhook():
    req = request.get_json(force=True)
    query_text = req.get('queryResult', {}).get('queryText', '')
    intent = req.get('queryResult', {}).get('intent', {}).get('displayName', 'default_session')

    state = intent
    action = agent.get_action(state)

    # Define when the chatbot "succeeds"
    if any(word in query_text.lower() for word in ["book", "schedule", "appointment", "talk", "call", "strategy", "meeting"]):
        reward = REWARD_SUCCESS
    else:
        reward = REWARD_FAILURE

    agent.update(state=state, action=action, reward=reward, next_state=state)

    return jsonify({
        'fulfillmentText': action
    })

if __name__ == '__main__':
    app.run(debug=True)