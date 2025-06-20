from agents.sentiment_agent import SentimentAgent, SentimentAgentInput
from agents.market_impact_agent import MarketImpactAgent, MarketImpactAgentInput
from agents.fact_check_agent import FactCheckAgent, FactCheckAgentInput
from evaluation.evaluator import Evaluator

test_articles = [
    {
        "article_id": "news_001",
        "headline": "TechCorp announces record Q3 earnings",
        "content": "TechCorp (NASDAQ: TECH) reported record Q3 earnings, beating analyst expectations.",
        "published_at": "2024-10-15T14:30:00Z"
    },
    {
        "article_id": "news_002",
        "headline": "AutoInc misses Q2 revenue targets",
        "content": "AutoInc (NYSE: AUTO) reported a loss for Q2, missing revenue targets.",
        "published_at": "2024-10-16T09:00:00Z"
    },
    {
        "article_id": "news_003",
        "headline": "Rumor: FinBank to acquire SmallBank",
        "content": "Unconfirmed sources suggest FinBank is in talks to acquire SmallBank.",
        "published_at": "2024-10-17T12:00:00Z"
    },
    {
        "article_id": "news_004",
        "headline": "RetailCo posts steady growth in Q4",
        "content": "RetailCo (NYSE: RET) reported steady growth in Q4, in line with analyst expectations.",
        "published_at": "2024-10-18T08:00:00Z"
    },
    {
        "article_id": "news_005",
        "headline": "EnergyPlus faces investigation over safety practices",
        "content": "EnergyPlus (NASDAQ: ENP) is under investigation for alleged safety violations.",
        "published_at": "2024-10-19T15:00:00Z"
    }
]

def analyze_article(article):
    sentiment_agent = SentimentAgent(use_llm=True)
    market_agent = MarketImpactAgent(use_llm=True)
    fact_agent = FactCheckAgent(use_llm=True)

    sentiment_result = sentiment_agent.analyze(SentimentAgentInput(**article))
    market_result = market_agent.analyze(MarketImpactAgentInput(**article))
    fact_result = fact_agent.analyze(FactCheckAgentInput(**article))

    return {
        "article_id": article["article_id"],
        "sentiment": sentiment_result.dict(),
        "market_impact": market_result.dict(),
        "fact_check": fact_result.dict()
    }

def batch_analyze(articles):
    results = []
    for article in articles:
        print(f"Analyzing {article['article_id']}...")
        results.append(analyze_article(article))
    return results

def pretty_print_result(result):
    print(f"\nArticle ID: {result['article_id']}")
    print("  Sentiment:")
    print(f"    Value: {result['sentiment']['sentiment']}")
    print(f"    Confidence: {result['sentiment']['confidence']}")
    print(f"    Rationale: {result['sentiment']['rationale']}")
    print("  Market Impact:")
    print(f"    Value: {result['market_impact']['impact']}")
    print(f"    Confidence: {result['market_impact']['confidence']}")
    print(f"    Rationale: {result['market_impact']['rationale']}")
    print("  Fact Check:")
    print(f"    Factual: {result['fact_check']['factual']}")
    print(f"    Confidence: {result['fact_check']['confidence']}")
    print(f"    Rationale: {result['fact_check']['rationale']}")

if __name__ == "__main__":
    results = batch_analyze(test_articles)
    print("\nAgent Results:")
    for r in results:
        pretty_print_result(r)
    evaluator = Evaluator()
    metrics = evaluator.evaluate(results, test_articles)
    print("\nEvaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")
