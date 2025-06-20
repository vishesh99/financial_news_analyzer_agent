from pydantic import BaseModel
from agents.gemini_utils import call_gemini
import json
import re

class MarketImpactAgentInput(BaseModel):
    article_id: str
    headline: str
    content: str
    published_at: str

class MarketImpactAgentOutput(BaseModel):
    impact: str
    confidence: float
    rationale: str

class MarketImpactAgent:
    def __init__(self, use_llm=True):
        self.use_llm = use_llm

    def analyze(self, input_data: MarketImpactAgentInput) -> MarketImpactAgentOutput:
        if self.use_llm:
            return self._analyze_with_llm(input_data)
        else:
            return self._analyze_with_rules(input_data)

    def _analyze_with_rules(self, input_data: MarketImpactAgentInput) -> MarketImpactAgentOutput:
        headline = input_data.headline.lower()
        if "record" in headline or "beats" in headline:
            return MarketImpactAgentOutput(
                impact="up",
                confidence=0.85,
                rationale="Positive earnings likely to move price up."
            )
        elif "misses" in headline or "loss" in headline:
            return MarketImpactAgentOutput(
                impact="down",
                confidence=0.85,
                rationale="Negative news likely to move price down."
            )
        else:
            return MarketImpactAgentOutput(
                impact="neutral",
                confidence=0.6,
                rationale="No clear market direction."
            )

    def _analyze_with_llm(self, input_data: MarketImpactAgentInput) -> MarketImpactAgentOutput:
        prompt = (
            "You are a financial market impact analysis expert.\n"
            "Given the following news article, predict the likely immediate market impact (up, down, or neutral), "
            "assign a confidence score between 0 and 1, and provide a brief rationale. "
            "Respond in JSON format with keys: impact, confidence, rationale.\n\n"
            f"Headline: {input_data.headline}\n"
            f"Content: {input_data.content}\n"
        )
        response = call_gemini(prompt)
        try:
            parsed = json.loads(response)
        except Exception:
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
            else:
                raise ValueError(f"Could not parse Gemini response: {response}")
        return MarketImpactAgentOutput(**parsed)
