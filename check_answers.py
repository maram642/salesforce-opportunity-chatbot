"""Vérifie que les réponses du chatbot correspondent aux données.

Deux modes :

  python check_answers.py
      Test automatique : pour CHAQUE opportunité et CHAQUE type de question,
      pose la question au chatbot et compare la réponse avec la valeur lue
      directement dans les données. Affiche le taux de réussite et les échecs.

  python check_answers.py --sample 5
      Contrôle manuel : tire 5 opportunités au hasard et affiche, pour chacune,
      la ligne brute des données à côté des réponses du bot. Vous comparez à l'oeil
      (ou avec le fichier CSV ouvert dans Excel).

Options : --seed 1 (autre tirage), --show-fails 20 (nombre d'échecs affichés)
"""
import argparse
import contextlib
import io
import random
import sys

import pandas as pd

with contextlib.redirect_stdout(io.StringIO()):  # masque le message "[data] Loaded ..."
    from salesforce_connector import load_data
from chatbot_core import Chatbot

# (clé, question, colonne des données, formatage de la valeur attendue, libellé affiché par le bot)
def _num(v):
    return f"{float(v):,.0f}"

CHECKS = [
    ("owner",        "Who owns {n}?",                      "Owner Full Name",     str,  "Owner"),
    ("stage",        "What stage is {n} in?",              "Stage",               str,  "Stage"),
    ("account",      "Which account is {n} for?",          "Account Name",        str,  "Account"),
    ("priority",     "What is the priority of {n}?",       "Actual Priority",     str,  "Priority"),
    ("sop",          "When does {n} start?",               "SOP",                 str,  "Start of production (SOP)"),
    ("eop",          "When does {n} end?",                 "EOP",                 str,  "End of production (EOP)"),
    ("dos",          "What is the decision date of {n}?",  "DOS",                 str,  "Decision date (DOS)"),
    ("sales",        "What are the total sales of {n}?",   "Total Sales",         _num, "Total Sales"),
    ("volume",       "What is the volume of {n}?",         "Total Volume",        _num, "Total Volume"),
    ("created",      "When was {n} created?",              "Created Date",        str,  "Created"),
    ("modified",     "When was {n} last modified?",        "Last Modified Date",  str,  "Last modified"),
    ("predecessor",  "What is the predecessor of {n}?",    "Predecessor Carline", str,  "Predecessor carline"),
]

MISSING_WORDS = ("not available", "none recorded")


def expected_text(value, fmt, column):
    """Ce que le chatbot doit afficher pour cette valeur."""
    if pd.isna(value) or str(value).strip() == "":
        return "none recorded" if column == "Predecessor Carline" else "not available"
    return fmt(value)


def run_checks(bot, df):
    results = []
    for _, row in df.iterrows():
        name = row["Opportunity Name"]
        for key, template, column, fmt, label in CHECKS:
            for variant in (name, name.rstrip("_")):  # avec et sans "__"
                question = template.format(n=variant)
                reply = bot.respond(question)
                expected = expected_text(row[column], fmt, column)
                # libellé + valeur exacts, et une seule ligne (pas d'autre opportunité ni de résumé)
                ok = f"{label}: {expected}" in reply and "\n" not in reply
                results.append({"opportunity": name, "check": key, "question": question,
                                "expected": expected, "reply": reply, "ok": ok})
    return pd.DataFrame(results)


def print_report(results, show_fails):
    total, passed = len(results), int(results["ok"].sum())
    print(f"\nRésultat : {passed}/{total} réponses correctes ({passed / total:.1%})\n")

    by_check = results.groupby("check")["ok"].agg(["sum", "count"])
    print(f"{'type de question':<14}{'corrects':>10}{'total':>8}")
    for key, r in by_check.iterrows():
        mark = "" if r["sum"] == r["count"] else "   <-- à regarder"
        print(f"{key:<14}{int(r['sum']):>10}{int(r['count']):>8}{mark}")

    fails = results[~results["ok"]]
    if fails.empty:
        print("\nAucune erreur : toutes les réponses correspondent aux données.")
        return 0
    print(f"\n{len(fails)} échec(s). Les {min(show_fails, len(fails))} premiers :")
    for _, f in fails.head(show_fails).iterrows():
        print(f"\n  Question : {f['question']}\n  Attendu  : {f['expected']}\n  Réponse  : {f['reply']!r}")
    return 1


def print_sample(bot, df, n, seed):
    rng = random.Random(seed)
    names = rng.sample(list(df["Opportunity Name"]), min(n, len(df)))
    for name in names:
        row = df[df["Opportunity Name"] == name].iloc[0]
        print("=" * 70)
        print(f"DONNÉES BRUTES pour {name}")
        for col in df.columns:
            print(f"  {col:<22}{row[col]}")
        print("\nRÉPONSES DU CHATBOT")
        for _, template, _, _, _ in CHECKS:
            q = template.format(n=name)
            print(f"  > {q}\n    {bot.respond(q).replace(chr(10), ' | ')}")
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=int, default=0, help="contrôle manuel sur N opportunités")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show-fails", type=int, default=10)
    args = parser.parse_args()

    with contextlib.redirect_stdout(io.StringIO()):
        df = load_data()
    bot = Chatbot(df)

    if args.sample:
        print_sample(bot, df, args.sample, args.seed)
        return 0

    print(f"Vérification de {len(df)} opportunités x {len(CHECKS)} types de questions x 2 écritures du nom...")
    results = run_checks(bot, df)
    return print_report(results, args.show_fails)


if __name__ == "__main__":
    sys.exit(main())
