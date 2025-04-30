
from flask import Flask, request, jsonify
from rl_agent import RLAgent

app = Flask(__name__)
agent = RLAgent(actions=[
    "Would you like to book a free 30-minute strategy call with us?",
    "Can I help schedule your free 30-minute session to grow your business?",
    "Want me to book a time to see how we attract ready-to-buy customers?",
    "Ready to book your complimentary 30-min strategy call?",
    "Would you like to reserve your free 30-min consultation to automate your leads?"
])

REWARD_SUCCESS = 1
REWARD_FAILURE = -1

@app.route('/webhook', methods=['POST'])
def webhook():
    query_text = request.json['queryResult']['queryText']
    session = request.json['session'].split('/')[-1]
    intent = request.json['queryResult']['intent']['displayName']

    state = intent
    action = agent.get_action(state)

    # simple rule: if the query contains a strong intent to book, give a reward
    if any(word in query_text.lower() for word in ["book", "schedule", "appointment", "call", "strategy", "meeting"]):
        agent.update(state, action, REWARD_SUCCESS, state)
    else:
        agent.update(state, action, REWARD_FAILURE, state)

    return jsonify({'fulfillmentText': action})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)