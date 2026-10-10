"""
Model 5: LSTM  (recurrent neural network)
-----------------------------------------
An LSTM reads demand as a sequence and keeps a memory of what it has seen, so in
theory it can learn patterns the tree models miss.

We tested it rather than assuming the tree models were best. On this dataset it
lost: there are only about 6,400 training hours, and demand here is driven mainly
by the calendar, which the tree models receive directly as features.

Needs TensorFlow (already installed in Google Colab).
Run:  python 05_lstm.py      (about 3 to 8 minutes)
"""

from rnn_common import train_network, tree_score_on_same_hours
from data_prep import report

predictions, actual, stamps = train_network(kind="LSTM")

lstm_mae = report("LSTM", actual, predictions)
tree_mae = tree_score_on_same_hours(stamps)

print(f"Gradient Boosting on the same {len(stamps):,} hours: MAE {tree_mae:.2f}")
if tree_mae < lstm_mae:
    print(f"The tree model is better by {lstm_mae - tree_mae:.2f} customers per hour.")
else:
    print(f"The LSTM is better by {tree_mae - lstm_mae:.2f} customers per hour.")
