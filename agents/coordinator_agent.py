from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from agents.agent_tools import sentiment_tool, market_impact_tool, fact_check_tool
import os

def coordinator_agent(article: dict):
    # Make sure your GEMINI_API_KEY is set in your environment or .env
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    tools = [sentiment_tool, market_impact_tool, fact_check_tool]
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    prompt = (
        f"Given the following financial news article, use the available tools to analyze sentiment, "
        f"market impact, and factual accuracy. If the tools disagree, discuss and try to reach a consensus. "
        f"Article: {article}"
    )
    return agent.run(prompt)
