# Salesforce Opportunity Chatbot

A chatbot that answers natural-language questions about Salesforce **Opportunity** records
("Who owns AM003?", "When does VW426 end?", "Give me a summary of BMWGEN6"), served through a
small Flask API so it can be embedded in a Salesforce page or any web front end.

Built during my internship at Draexlmaier against their Salesforce sandbox (268 automotive
carline opportunities, 13 fields). This public version runs on a **synthetic dataset with the
same schema**, because the original data is confidential and the sandbox access has ended.

## How it works

```
user message
   ├─ greeting?                       -> friendly reply
   ├─ find opportunity name(s)        -> exact match ("AM003" == "AM003__"), fuzzy suggestions, conversation memory
   ├─ classify intent(s)              -> keyword scoring (default) or zero-shot BART model (optional)
   └─ look up values in a DataFrame   -> formatted answer
```

| File | Role |
|---|---|
| `salesforce_connector.py` | SOQL query via `simple-salesforce`, flattening of `Account.Name` / `Owner.Name`, renaming, date formatting, **CSV fallback** (`load_data()`) |
| `chatbot_core.py` | The `Chatbot` class: entity matching, intent classification, answers, memory |
| `app.py` | Flask app: serves the chat page at `/`, the API at `POST /chat`, status at `GET /health` |
| `static/index.html` | The chat web interface (plain HTML/CSS/JS, no build step, light and dark mode) |
| `generate_sample_data.py` | Creates `data/opportunities.csv` (fake data, same 13 columns) |
| `test_chatbot.py` | 21 tests, including the connector with a fake Salesforce response |
| `check_answers.py` | Automatically checks every chatbot answer against the underlying data (see below) |


## Quick start

```bash
pip install -r requirements.txt
python generate_sample_data.py      # creates data/opportunities.csv
python app.py                       # then open http://127.0.0.1:5000 in your browser
```

The page shows whether it is running on **live Salesforce data** or the **demo CSV**, and
suggests questions using real opportunity names from the data.

```bash
curl -X POST http://127.0.0.1:5000/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "Who owns AM003?"}'
```

Run the tests: `python -m unittest -v`

## Checking the chatbot's answers

`check_answers.py` verifies that the chatbot's answers match the data. It has two modes:

```bash
python check_answers.py
```

**Automatic test.** For every opportunity and each of the 12 question types (owner, stage,
account, priority, SOP, EOP, DOS, total sales, total volume, created, last modified,
predecessor), it asks the chatbot and compares the reply with the value read directly from the
data. Each name is tested with and without the trailing `__`. It prints the success rate per
question type and the first failures.

```bash
python check_answers.py --sample 5
```

**Manual check.** Picks 5 random opportunities and prints the raw data row next to the
chatbot's answers, so you can compare them by eye.

Options: `--seed 1` (different random draw), `--show-fails 20` (number of failures displayed).
The script exits with a non-zero code if any answer is wrong, so it can also run in CI.

## Connecting to a real Salesforce org

1. Copy `.env.example` to `.env` and fill in `SF_USERNAME`, `SF_PASSWORD`, `SF_TOKEN`
   (`SF_DOMAIN=test` for a sandbox, `login` for production or a free Developer Edition org).
2. Make sure the org has these Opportunity fields (API names):
   `Actual_Priority__c`, `SOP__c`, `EOP__c`, `Expected_DOS__c`, `Total_Sales__c`,
   `Total_Volume__c`, `Predecessor_Carline__c`.
3. Run `python app.py`. With credentials present the app reads from Salesforce, otherwise from the CSV.
   Force one or the other with `DATA_SOURCE=salesforce` or `DATA_SOURCE=csv`.

Credentials are read from environment variables only; nothing sensitive is stored in the code.

## Optional: zero-shot intent model

`pip install transformers torch` and set `USE_ZERO_SHOT=1` to classify intents with
`facebook/bart-large-mnli` (about 1.6 GB download); the keyword rules remain as the fallback
when the model is unsure. The default rule-based mode needs no downloads and handles all the
intents in the test suite.

## Supported questions

Owner, stage, account, priority, SOP (start), EOP (end), DOS (decision date), total sales, total
volume, predecessor carline, created / last-modified dates, and a full summary. Follow-ups
("and its sales?") use the last opportunity mentioned, and one message can ask several things
("when does VW426 end and who owns it").