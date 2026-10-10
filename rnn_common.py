"""
rnn_common.py
-------------
Shared helper for the two neural network scripts (LSTM and GRU).

A recurrent network does not read a table of features like the tree models do.
It reads a SEQUENCE of past demand. For every hour we want to predict, we give it:

  * a window of the previous 14 days of hourly demand, ending 7 days before the
    hour being predicted (the same week-ahead rule as the other models), and
  * the calendar information for the hour itself (hour, weekday, month).

Because the network needs 21 days of history before it can predict anything, it
loses the first rows. The classical models are therefore re-scored on exactly
the same remaining hours, so the comparison stays fair.
"""

import numpy as np
import pandas as pd
from data_prep import load_data, OPEN_HOUR, CLOSE_HOUR, SPLIT_DATE, FEATURES, TARGET

HOURS_PER_DAY = CLOSE_HOUR - OPEN_HOUR + 1
WINDOW = 14 * HOURS_PER_DAY   # the network reads 14 days of history
GAP = 7 * HOURS_PER_DAY       # ending one week before the target hour


def build_sequences():
    """Return sequences, calendar inputs, targets, timestamps and the split masks."""
    train, test, _, _ = load_data(verbose=False)
    full = pd.concat([train, test]).sort_index()

    values = full[TARGET].values.astype("float32")
    calendar = pd.get_dummies(
        full[["hour", "day_number", "month"]].astype(str)
    ).values.astype("float32")

    sequences, calendars, targets, stamps = [], [], [], []
    for i in range(WINDOW + GAP, len(full)):
        sequences.append(values[i - WINDOW - GAP:i - GAP])
        calendars.append(calendar[i])
        targets.append(values[i])
        stamps.append(full.index[i])

    sequences = np.array(sequences)[..., None]     # shape: (rows, 182, 1)
    calendars = np.array(calendars)
    targets = np.array(targets)
    stamps = pd.DatetimeIndex(stamps)

    is_train = stamps < SPLIT_DATE
    is_test = ~is_train
    return sequences, calendars, targets, stamps, is_train, is_test


def train_network(kind="LSTM", units=64, epochs=60):
    """Build, train and evaluate a recurrent network. Returns predictions and truth."""
    import tensorflow as tf
    tf.random.set_seed(42)

    seqs, cals, y, stamps, is_train, is_test = build_sequences()
    scale = y[is_train].max()          # scale demand into roughly 0 to 1

    layer = tf.keras.layers.LSTM if kind.upper() == "LSTM" else tf.keras.layers.GRU

    seq_input = tf.keras.Input(seqs.shape[1:], name="demand_history")
    cal_input = tf.keras.Input((cals.shape[1],), name="calendar")
    x = layer(units)(seq_input / scale)                 # read the sequence
    x = tf.keras.layers.Concatenate()([x, cal_input])   # add the calendar facts
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    output = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model([seq_input, cal_input], output)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mae")
    print(f"Training the {kind}. This takes a few minutes on a normal CPU.\n")

    model.fit(
        [seqs[is_train], cals[is_train]], y[is_train] / scale,
        validation_split=0.15, epochs=epochs, batch_size=64,
        verbose=0, shuffle=False,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)],
    )

    predictions = model.predict([seqs[is_test], cals[is_test]], verbose=0).ravel() * scale
    return predictions, y[is_test], stamps[is_test]


def tree_score_on_same_hours(stamps):
    """Score Gradient Boosting on the identical hours, so the comparison is fair."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error

    train, test, features, target = load_data(verbose=False)
    model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=42)
    model.fit(train[features], train[target])

    same = test[test.index.isin(stamps)]
    predicted = np.clip(model.predict(same[features]), 0, None)
    return mean_absolute_error(same[target], predicted)
