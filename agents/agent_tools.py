from langchain.tools import Tool
from agents.sentiment_agent import SentimentAgent, SentimentAgentInput
from agents.market_impact_agent import MarketImpactAgent, MarketImpactAgentInput
from agents.fact_check_agent import FactCheckAgent, FactCheckAgentInput
import ast

def sentiment_tool(article):
    if isinstance(article, str):
        article = ast.literal_eval(article)
    agent = SentimentAgent(use_llm=True)
    result = agent.analyze(SentimentAgentInput(**article))
    return str(result.dict())

def market_impact_tool(article):
    if isinstance(article, str):
        article = ast.literal_eval(article)
    agent = MarketImpactAgent(use_llm=True)
    result = agent.analyze(MarketImpactAgentInput(**article))
    return str(result.dict())

def fact_check_tool(article):
    if isinstance(article, str):
        article = ast.literal_eval(article)
    agent = FactCheckAgent(use_llm=True)
    result = agent.analyze(FactCheckAgentInput(**article))
    return str(result.dict())

sentiment_tool = Tool(
    name="SentimentAgent",
    func=sentiment_tool,
    description="Analyzes the sentiment of a financial news article. Input is a dict with keys: article_id, headline, content, published_at."
)

market_impact_tool = Tool(
    name="MarketImpactAgent",
    func=market_impact_tool,
    description="Assesses the likely market impact of a financial news article. Input is a dict with keys: article_id, headline, content, published_at."
)

fact_check_tool = Tool(
    name="FactCheckAgent",
    func=fact_check_tool,
    description="Checks the factual accuracy of a financial news article. Input is a dict with keys: article_id, headline, content, published_at."
)
