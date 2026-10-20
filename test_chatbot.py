"""Tests: run with `python -m unittest -v` (no extra packages needed).

Covers the chatbot on the sample CSV, and the Salesforce connector's cleaning
logic using a fake Salesforce object shaped like a real query_all() response.
"""
import os
import re
import unittest

import pandas as pd

import salesforce_connector as sc
from chatbot_core import Chatbot, greet


class FakeSalesforce:
    """Mimics simple_salesforce.Salesforce.query_all() for two opportunities."""

    def query_all(self, query):
        assert "FROM Opportunity" in query
        return {"records": [
            {
                "attributes": {"type": "Opportunity"},
                "Name": "AM003__",
                "Account": {"attributes": {}, "Name": "Aston Martin"},
                "Owner": {"attributes": {}, "Name": "Jane Doe"},
                "StageName": "Closed Won",
                "Actual_Priority__c": "1",
                "SOP__c": "2025-07-01",
                "EOP__c": "2027-01-01",
                "Expected_DOS__c": None,
                "CreatedDate": "2025-06-19T17:03:10.000+0000",
                "LastModifiedDate": "2025-06-19T17:03:11.000+0000",
                "Total_Sales__c": 1582.5,
                "Total_Volume__c": 15825.0,
                "Predecessor_Carline__c": "a039Q000012ZhJpQAK",
            },
            {
                "attributes": {"type": "Opportunity"},
                "Name": "VW426__",
                "Account": {"attributes": {}, "Name": "Volkswagen"},
                "Owner": {"attributes": {}, "Name": "John Roe"},
                "StageName": "Draft Phase",
                "Actual_Priority__c": "3",
                "SOP__c": "2026-02-01",
                "EOP__c": "2033-03-01",
                "Expected_DOS__c": "2024-02-23",
                "CreatedDate": "2025-06-19T17:00:55.000+0000",
                "LastModifiedDate": "2025-07-09T09:27:06.000+0000",
                "Total_Sales__c": 858343300.0,
                "Total_Volume__c": 994700.0,
                "Predecessor_Carline__c": None,
            },
        ]}


class ConnectorTests(unittest.TestCase):
    def test_salesforce_records_are_cleaned(self):
        df = sc.fetch_opportunities(FakeSalesforce())
        self.assertEqual(set(sc.EXPECTED_COLUMNS), set(df.columns))
        row = df[df["Opportunity Name"] == "AM003__"].iloc[0]
        self.assertEqual(row["Account Name"], "Aston Martin")
        self.assertEqual(row["Owner Full Name"], "Jane Doe")
        self.assertEqual(row["SOP"], "01.07.2025")
        # 17:03 UTC in June = 19:03 in Paris (CEST)
        self.assertEqual(row["Created Date"], "19.06.2025, 19:03")
        self.assertTrue(pd.isna(df[df["Opportunity Name"] == "AM003__"]["DOS"].iloc[0]))

    def test_chatbot_works_on_salesforce_shaped_data(self):
        bot = Chatbot(sc.fetch_opportunities(FakeSalesforce()))
        self.assertIn("Jane Doe", bot.respond("who owns AM003?"))
        self.assertIn("not available", bot.respond("decision date of AM003"))

    def test_csv_mode_without_credentials(self):
        for k in ("SF_USERNAME", "SF_PASSWORD", "SF_TOKEN", "DATA_SOURCE"):
            os.environ.pop(k, None)
        df = sc.load_data()
        self.assertEqual(list(df.columns), sc.EXPECTED_COLUMNS)
        self.assertGreater(len(df), 50)

    def test_salesforce_mode_requires_credentials(self):
        for k in ("SF_USERNAME", "SF_PASSWORD", "SF_TOKEN"):
            os.environ.pop(k, None)
        with self.assertRaises(RuntimeError):
            sc.get_salesforce_connection()



    def test_no_hardcoded_secrets(self):
      import re
      with open(sc.__file__, encoding="utf-8") as f:
        source = f.read()
    # aucune affectation littérale de mot de passe / token
      self.assertIsNone(re.search(r"(password|security_token)\s*=\s*[\"'][^\"']+[\"']", source))        


class ChatbotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for k in ("SF_USERNAME", "SF_PASSWORD", "SF_TOKEN", "DATA_SOURCE"):
            os.environ.pop(k, None)
        cls.df = sc.load_data()
        cls.bot = Chatbot(cls.df)

    def setUp(self):
        self.bot.memory = {"last_opportunity": None, "last_intent": None}

    def value(self, name, col):
        return self.df[self.df["Opportunity Name"] == name].iloc[0][col]

    def test_greeting(self):
        self.assertIsNotNone(greet("hi"))
        self.assertIsNotNone(greet("Good morning!"))
        self.assertIsNone(greet("what stage is AM003 in"))
        self.assertIsNone(greet("hi who owns AM003, thanks a lot"))

    def test_the_original_example_questions(self):
        b = self.bot
        self.assertIn(self.value("AM003__", "SOP"), b.respond("What is the start date of AM003?"))
        self.assertIn(self.value("AM003__", "EOP"), b.respond("When does AM003 project end?"))
        self.assertIn(self.value("AM003__", "Stage"), b.respond("What stage is AM003 opportunity in?"))
        self.assertIn(self.value("AM003__", "Owner Full Name"), b.respond("Tell me who is responsible for AM003."))
        self.assertIn(self.value("AM003__", "Account Name"), b.respond("Which account is AM003 for?"))
        self.assertIn(str(self.value("AM003__", "Actual Priority")), b.respond("Show me priority of AM003 opportunity"))
        self.assertIn(self.value("AM003__", "Last Modified Date"), b.respond("last modified date of AM003__ project?"))

    def test_summary(self):
        out = self.bot.respond("Give me a summary for Opportunity AM003.")
        for part in ("Summary for AM003__", "Owner", "Stage", "Account", "Total Sales", "Total Volume"):
            self.assertIn(part, out)

    def test_each_intent(self):
        b = self.bot
        cases = {
            "who owns BMWGEN6": "Owner",
            "what is the decision date of BMWGEN6": "Decision date",
            "what is the revenue of BMWGEN6": "Total Sales",
            "what is the volume of BMWGEN6": "Total Volume",
            "when was BMWGEN6 created": "Created",
            "what is the predecessor of BMWGEN6": "Predecessor",
            "when does BMWGEN6 start": "SOP",
            "when does BMWGEN6 finish": "EOP",
        }
        for question, expected in cases.items():
            with self.subTest(question):
                self.assertIn(expected, b.respond(question))

    def test_name_with_or_without_trailing_underscores_and_case(self):
        a = self.bot.respond("who owns am003")
        b = self.bot.respond("WHO OWNS AM003__")
        self.assertEqual(a, b)

    def test_follow_up_uses_memory(self):
        self.bot.respond("who owns AM003?")
        out = self.bot.respond("and what is its stage?")
        self.assertIn(self.value("AM003__", "Stage"), out)

    def test_two_opportunities_in_one_question(self):
        out = self.bot.respond("who owns AM003 and VW426?")
        self.assertIn(self.value("AM003__", "Owner Full Name"), out)
        self.assertIn(self.value("VW426__", "Owner Full Name"), out)

    def test_unknown_opportunity_is_not_guessed(self):
        out = self.bot.respond("who owns ZZZ999?")
        self.assertNotIn("Owner:", out)
        self.assertIn("ZZZ999", out)

    def test_unknown_opportunity_does_not_fall_back_to_memory(self):
        self.bot.respond("who owns AM003?")
        out = self.bot.respond("who owns ZZZ999?")
        self.assertNotIn(self.value("AM003__", "Owner Full Name"), out)
        self.assertIn("couldn't find", out)

    def test_two_questions_in_one_message(self):
        out = self.bot.respond("when does VW426 end and who owns it")
        self.assertIn(self.value("VW426__", "EOP"), out)
        self.assertIn(self.value("VW426__", "Owner Full Name"), out)

    def test_tell_me_about_the_owner_is_not_a_full_summary(self):
        out = self.bot.respond("tell me about the owner of AM003")
        self.assertIn("Owner", out)
        self.assertNotIn("Summary for", out)

    def test_message_without_name_or_memory(self):
        self.assertIn("couldn't find", self.bot.respond("who owns it"))

    def test_gibberish_and_empty(self):
        self.assertIn("Ask me about", self.bot.respond("blah blah"))
        self.assertIn("Ask me about", self.bot.respond("   "))

    def test_name_only_gives_summary(self):
        self.assertIn("Summary for AM003__", self.bot.respond("AM003"))

    def test_partial_name_asks_for_clarification(self):
        names = [n for n in self.df["Opportunity Name"] if "/" in n]
        if not names:
            self.skipTest("no slash-style names in the sample data")
        stem = names[0].split("/")[0].lower()
        out = self.bot.respond(f"who owns {stem}")
        self.assertTrue("Owner" in out or "Which one" in out or "Assuming" in out)


class FlaskTests(unittest.TestCase):
    def test_endpoints(self):
        try:
            import app as app_module
        except ImportError:
            self.skipTest("flask / flask_cors not installed")
        client = app_module.app.test_client()
        self.assertEqual(client.get("/health").status_code, 200)
        r = client.post("/chat", json={"message": "who owns AM003?"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("Owner", r.get_json()["reply"])
        self.assertEqual(client.post("/chat", json={}).status_code, 400)

        page = client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"Opportunity Assistant", page.data)
        health = client.get("/health").get_json()
        self.assertEqual(health["source"], "csv")
        self.assertEqual(len(health["examples"]), 3)
        self.assertEqual(client.post("/chat", data="not json").status_code, 400)


if __name__ == "__main__":
    unittest.main()
