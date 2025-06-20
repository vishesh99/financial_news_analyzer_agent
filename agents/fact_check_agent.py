from pydantic import BaseModel
from agents.gemini_utils import call_gemini
import json
import re

class FactCheckAgentInput(BaseModel):
    article_id: str
    headline: str
    content: str
    published_at: str

class FactCheckAgentOutput(BaseModel):
    factual: bool
    confidence: float
    rationale: str

class FactCheckAgent:
    def __init__(self, use_llm=True):
        self.use_llm = use_llm

    def analyze(self, input_data: FactCheckAgentInput) -> FactCheckAgentOutput:
        if self.use_llm:
            return self._analyze_with_llm(input_data)
        else:
            return self._analyze_with_rules(input_data)

    def _analyze_with_rules(self, input_data: FactCheckAgentInput) -> FactCheckAgentOutput:
        content = input_data.content.lower()
        if "rumor" in content or "unconfirmed" in content:
            return FactCheckAgentOutput(
                factual=False,
                confidence=0.8,
                rationale="Article contains terms indicating uncertainty."
            )
        else:
            return FactCheckAgentOutput(
                factual=True,
                confidence=0.9,
                rationale="No evidence of inaccuracy detected."
            )

    def _analyze_with_llm(self, input_data: FactCheckAgentInput) -> FactCheckAgentOutput:
        prompt = (
            "You are a financial news fact-checking expert.\n"
            "Given the following news article, determine if the information is factual (true/false), "
            "assign a confidence score between 0 and 1, and provide a brief rationale. "
            "Respond in JSON format with keys: factual (true/false), confidence, rationale.\n\n"
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
        return FactCheckAgentOutput(**parsed)
