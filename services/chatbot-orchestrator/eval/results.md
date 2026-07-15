# chatbot-orchestrator eval results

Router model: `gemini-2.5-flash-lite`. Tool-selection accuracy over 10 messages (downstream services stubbed -- this measures routing only).

| metric | value |
|---|---|
| tool-selection accuracy | 10/10 = 1.00 |

| id | message | expected | picked | ok |
|---|---|---|---|---|
| o01 | How fast did Meridian's data center segment grow? | query_financial_documents | query_financial_documents | ✅ |
| o02 | What full-year installation guidance did Helios Energy provi | query_financial_documents | query_financial_documents | ✅ |
| o03 | What were Northwind's comparable store sales in Q3 fiscal 20 | query_financial_documents | query_financial_documents | ✅ |
| o04 | Summarize Meridian's foundry dependence risk from its 10-K. | query_financial_documents | query_financial_documents | ✅ |
| o05 | Classify this headline: Northwind beat estimates and raised  | classify_financial_news | classify_financial_news | ✅ |
| o06 | Is this bullish or bearish: regulators opened an investigati | classify_financial_news | classify_financial_news | ✅ |
| o07 | What's the sentiment and topic of: Aster Pharma agreed to bu | classify_financial_news | classify_financial_news | ✅ |
| o08 | Hello, what can you help me with? | none | none | ✅ |
| o09 | Should I buy Meridian stock right now? | none | none | ✅ |
| o10 | What's the weather like today? | none | none | ✅ |
