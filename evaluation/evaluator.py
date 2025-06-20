from typing import List, Dict

class Evaluator:
    def __init__(self):
        pass

    def agent_agreement_rate(self, results: List[Dict]) -> float:
        agree = 0
        for r in results:
            s = r['sentiment']['sentiment'].lower()
            m = r['market_impact']['impact'].lower()
            if (s == "positive" and m == "up") or (s == "negative" and m == "down") or (s == "neutral" and m == "neutral"):
                agree += 1
        return agree / len(results) if results else 0.0

    def confidence_spread(self, results: List[Dict]) -> float:
        spreads = []
        for r in results:
            confidences = [
                r['sentiment']['confidence'],
                r['market_impact']['confidence'],
                r['fact_check']['confidence']
            ]
            spreads.append(max(confidences) - min(confidences))
        return sum(spreads) / len(spreads) if spreads else 0.0

    def consistency(self, results: List[Dict], articles: List[Dict]) -> float:
        from collections import defaultdict
        groups = defaultdict(list)
        for r, a in zip(results, articles):
            key = a['headline'].split()[0].lower()
            groups[key].append(r['sentiment']['sentiment'].lower())
        consistent = 0
        for group in groups.values():
            if len(set(group)) == 1:
                consistent += 1
        return consistent / len(groups) if groups else 1.0

    def evaluate(self, results: List[Dict], articles: List[Dict]) -> Dict:
        return {
            "agreement_rate": self.agent_agreement_rate(results),
            "confidence_spread": self.confidence_spread(results),
            "consistency": self.consistency(results, articles)
        }
