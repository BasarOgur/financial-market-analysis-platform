# market-data-service eval results

Live yfinance lookups against 7 known tickers (mix of valid and invalid).

| metric | value |
|---|---|
| correctness | 7/7 = 1.00 |

| id | ticker | expected found | detail | ok |
|---|---|---|---|---|
| m01 | AAPL | True | price=314.86 | ✅ |
| m02 | MSFT | True | price=384.93 | ✅ |
| m03 | GOOGL | True | price=359.51 | ✅ |
| m04 | AMZN | True | price=247.49 | ✅ |
| m05 | TSLA | True | price=396.18 | ✅ |
| m06 | NOTREAL1 | False | not found | ✅ |
| m07 | ZZZFAKE | False | not found | ✅ |
