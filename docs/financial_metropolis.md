# Financial Metropolis & University

The nine-core simulation now has a civic layer and a market-research university. All accounts, grants, taxes and decisions in this layer are virtual. The system does not submit real trades.

## City institutions

- **Elections and council:** eligible cores vote periodically; the winner and three-member council are recorded in an auditable history.
- **Police / integrity audit:** checks narrow software invariants and records critical findings. It cannot rewrite scores, votes, or trade ledgers.
- **Court and appeals:** temporary suspension requires repeated critical findings; expiry triggers a review and possible reinstatement.
- **Legislature and constitution:** publishes policy records. The paper-only rule, risk veto and audit requirements are not user-overridable by simulated votes.
- **Virtual treasury:** records a small simulated levy on positive closed paper-trade P&L. It is not real money.
- **Market ecology:** records repeated project selections and warns when candidate selection becomes concentrated.
- **Public bulletin:** retains a short audit trail of civic events.

## University of Financial Markets

Eight departments are initialized:

1. Quantitative finance, statistics and econometrics
2. Price action and market structure
3. Risk management and drawdown control
4. On-chain data, liquidity and volume quality
5. News, macroeconomics and event studies
6. Machine learning, overfitting and drift
7. Portfolio theory, correlation and diversification
8. Market microstructure, slippage and execution costs

Each core is a student with a major, minor, skills ledger, transcript, credits and GPA-like skill average. Faculty/departments provide deterministic course curricula. Closed paper trades become case studies; a single win/loss changes skills only slightly and is explicitly not treated as causal proof. Research reports remain drafts, and grants are virtual.

### How education affects market analysis

The university applies a bounded adjustment of at most **+2 or -2 points** to a core's candidate score based on its trained domain and the candidate's existing strategy evidence. It cannot bypass market-data validation, risk vetoes, core eligibility, or the 5-of-9 quorum. Education is a ranking refinement, not a claim that the bot has achieved reliable predictive intelligence.

## Validation

Run:

```bash
python -m py_compile market_university.py core_civilization.py nine_core_engine.py
python -m unittest tests.test_market_university tests.test_core_civilization tests.test_nine_core_engine -v
```

GitHub Actions also compiles the university module and runs the repository regression suite.

## Important limitations

- Course progress is currently a deterministic simulation, not external university-grade training or a large-language-model fine-tuning pipeline.
- Market research reports summarize paper-trading observations and do not prove causal relationships.
- University skills and rankings must be evaluated with chronological out-of-sample tests, realistic fees/slippage, and enough resolved outcomes before trusting any change.
- No part of this city places real buy/sell orders or guarantees returns.
