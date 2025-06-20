from pydantic import BaseModel
from agents.gemini_utils import call_gemini
import json
import re

class SentimentAgentInput(BaseModel):
    article_id: str
    headline: str
    content: str
    published_at: str

class SentimentAgentOutput(BaseModel):
    sentiment: str
    confidence: float
    rationale: str

class SentimentAgent:
    def __init__(self, use_llm=True):
        self.use_llm = use_llm

    def analyze(self, input_data: SentimentAgentInput) -> SentimentAgentOutput:
        if self.use_llm:
            return self._analyze_with_llm(input_data)
        else:
            return self._analyze_with_rules(input_data)

    def _analyze_with_rules(self, input_data: SentimentAgentInput) -> SentimentAgentOutput:
        headline = input_data.headline.lower()
        if "record" in headline or "beats" in headline:
            return SentimentAgentOutput(
                sentiment="positive",
                confidence=0.9,
                rationale="Headline contains strong positive financial terms."
            )
        elif "misses" in headline or "loss" in headline:
            return SentimentAgentOutput(
                sentiment="negative",
                confidence=0.9,
                rationale="Headline contains negative financial terms."
            )
        else:
            return SentimentAgentOutput(
                sentiment="neutral",
                confidence=0.6,
                rationale="No strong sentiment detected."
            )

    def _analyze_with_llm(self, input_data: SentimentAgentInput) -> SentimentAgentOutput:
        prompt = (
            "You are a financial news sentiment analysis expert.\n"
            "Given the following news article, analyze the overall sentiment (positive, negative, or neutral), "
            "assign a confidence score between 0 and 1, and provide a brief rationale. "
            "Respond in JSON format with keys: sentiment, confidence, rationale.\n\n"
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
        return SentimentAgentOutput(**parsed)