"""
LLM Fraud Explainability Agent (LangChain + OpenAI GPT-4).

Takes SHAP values from the ensemble and generates human-readable
natural-language fraud justifications for flagged transactions.
"""
from __future__ import annotations


from app.core.config import settings


SYSTEM_PROMPT = """You are a fraud analysis expert at a financial institution.
You receive a flagged transaction and its top SHAP feature contributions,
which explain why the model scored it as potentially fraudulent.

Your job is to write a clear, concise fraud justification (2-4 sentences) for
a fraud analyst. Be specific about which features drove the decision and why
they indicate fraud risk. Do not use technical ML jargon — explain in business terms.

Always end with a recommended action: BLOCK, REVIEW, or MONITOR."""

ANALYSIS_PROMPT = """Transaction Details:
- Transaction ID: {transaction_id}
- Amount: ${amount:.2f}
- Product Category: {product_cd}
- Card: ending in {card_last4}
- Fraud Probability Score: {fraud_score:.1%}

Top Risk Factors (SHAP analysis):
{shap_summary}

AutoEncoder Anomaly Score: {anomaly_score:.4f} (threshold: {ae_threshold:.4f})

Generate a fraud justification and recommended action."""


def format_shap_summary(top_features: list[dict]) -> str:
    lines = []
    for i, feat in enumerate(top_features[:7], 1):
        direction = "↑ increases" if feat["shap_value"] > 0 else "↓ decreases"
        lines.append(
            f"  {i}. {feat['feature']}: {direction} fraud risk "
            f"(SHAP={feat['shap_value']:+.4f})"
        )
    return "\n".join(lines)


class FraudExplainabilityAgent:
    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.1,
        api_key: str | None = None,
    ):
        from langchain.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        self.llm = ChatOpenAI(
            model=model or settings.openai_model,
            temperature=temperature,
            api_key=api_key or settings.openai_api_key,
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", ANALYSIS_PROMPT),
            ]
        )
        self.chain = self.prompt | self.llm

    def explain(
        self,
        transaction: dict,
        shap_explanation: dict,
        anomaly_score: float,
        ae_threshold: float,
        fraud_score: float,
    ) -> str:
        top_features = shap_explanation.get("top_features", [])
        shap_summary = format_shap_summary(top_features)

        response = self.chain.invoke(
            {
                "transaction_id": transaction.get("TransactionID", "N/A"),
                "amount": float(transaction.get("TransactionAmt", 0)),
                "product_cd": transaction.get("ProductCD", "Unknown"),
                "card_last4": str(transaction.get("card1", "****"))[-4:],
                "fraud_score": fraud_score,
                "shap_summary": shap_summary,
                "anomaly_score": anomaly_score,
                "ae_threshold": ae_threshold,
            }
        )
        return response.content

    def batch_explain(
        self,
        transactions: list[dict],
        shap_explanations: list[dict],
        anomaly_scores: list[float],
        ae_threshold: float,
        fraud_scores: list[float],
    ) -> list[str]:
        return [
            self.explain(tx, shap_exp, ae_score, ae_threshold, fs)
            for tx, shap_exp, ae_score, fs in zip(
                transactions, shap_explanations, anomaly_scores, fraud_scores
            )
        ]


class RuleBasedExplainer:
    RISK_TEMPLATES = {
        "velocity_count_1h": "High transaction velocity ({val:.0f} transactions in last hour)",
        "velocity_sum_24h": "Unusually high 24h spend (${val:.2f})",
        "addr_distance_km": "Large billing/shipping distance ({val:.0f}km)",
        "ae_anomaly_score": "Transaction pattern deviates significantly from card history",
        "merchant_fraud_rate": "High-risk merchant category (fraud rate: {val:.1%})",
        "amt_zscore": "Amount is {val:.1f} standard deviations above card average",
        "tx_is_night": "Transaction occurred during off-hours",
        "addr_mismatch": "Billing and shipping addresses do not match",
        "high_risk_merchant": "Merchant category flagged as high-risk",
    }

    def explain(
        self,
        transaction: dict,
        shap_explanation: dict,
        anomaly_score: float,
        ae_threshold: float,
        fraud_score: float,
    ) -> str:
        top = shap_explanation.get("top_features", [])[:3]
        reasons = []
        for feat in top:
            if feat["shap_value"] > 0:
                key = feat["feature"]
                template = self.RISK_TEMPLATES.get(key, f"Elevated {key} risk factor")
                reasons.append(template)

        reason_str = "; ".join(reasons) if reasons else "Multiple risk factors detected"
        action = "BLOCK" if fraud_score > 0.8 else "REVIEW" if fraud_score > 0.5 else "MONITOR"

        return (
            f"Transaction ${transaction.get('TransactionAmt', 0):.2f} flagged "
            f"with {fraud_score:.1%} fraud probability. "
            f"Key indicators: {reason_str}. "
            f"Recommended action: {action}."
        )


def get_explainer(use_llm: bool = True) -> FraudExplainabilityAgent | RuleBasedExplainer:
    if use_llm and settings.openai_api_key and settings.openai_api_key != "sk-placeholder":
        return FraudExplainabilityAgent()
    return RuleBasedExplainer()
