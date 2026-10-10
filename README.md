# LSTM Recurrent Neural Network

Long Short-Term Memory (LSTM) sequence model trained on sequential demand history.

## Files Included
- `05_lstm.py` (also accessible as `main.py`): Model execution script
- `data_prep.py`: Data cleaning, hourly aggregation, feature engineering, and evaluation
- `Bakery sales.csv`: Raw transaction dataset
- `requirements.txt`: Python package dependencies

## Setup and Execution

1. **Create and activate a virtual environment:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the model:**
```bash
python 05_lstm.py
# or: python main.py
```

Generated forecast plots are saved to the `outputs/` folder upon execution.
