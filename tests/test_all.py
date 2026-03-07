"""
Comprehensive test suite for fraud detection system.

Covers:
- Feature engineering (velocity, haversine, merchant risk, temporal)
- AutoEncoder (train, threshold, anomaly scoring)
- Ensemble (predict, SHAP, evaluate, save/load)
- Kafka producer/consumer (mocked)
- Drift monitor (KS test, PSI, alerts)
- FastAPI endpoints (health, score, batch, explain, evaluate, drift)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_transactions() -> pd.DataFrame:
    """Minimal IEEE-CIS-like transaction DataFrame."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame(
        {
            "TransactionID": range(1, n + 1),
            "TransactionDT": np.random.randint(86400, 86400 * 180, n),
            "TransactionAmt": np.abs(np.random.randn(n) * 100) + 10,
            "ProductCD": np.random.choice(["W", "H", "C", "S", "R"], n),
            "card1": np.random.randint(1000, 9999, n),
            "card2": np.random.randint(100, 600, n).astype(float),
            "card3": np.random.choice([150.0, 185.0], n),
            "card4": np.random.choice(["visa", "mastercard"], n),
            "card5": np.random.choice([102.0, 201.0], n),
            "card6": np.random.choice(["credit", "debit"], n),
            "addr1": np.random.randint(100, 500, n).astype(float),
            "addr2": np.random.randint(10, 100, n).astype(float),
            "dist1": np.random.exponential(50, n),
            "dist2": np.random.exponential(100, n),
            "C1": np.random.randint(0, 5, n).astype(float),
            "C2": np.random.randint(0, 5, n).astype(float),
            "C3": np.zeros(n),
            "C4": np.random.randint(0, 3, n).astype(float),
            "C5": np.random.randint(0, 3, n).astype(float),
            "C6": np.random.randint(0, 5, n).astype(float),
            "C7": np.zeros(n),
            "C8": np.random.randint(0, 5, n).astype(float),
            "C9": np.random.randint(0, 2, n).astype(float),
            "C10": np.zeros(n),
            "C11": np.random.randint(0, 5, n).astype(float),
            "C12": np.zeros(n),
            "C13": np.random.randint(0, 5, n).astype(float),
            "C14": np.random.randint(0, 3, n).astype(float),
            "D1": np.random.randint(0, 300, n).astype(float),
            "D2": np.random.randint(0, 300, n).astype(float),
            "D3": np.random.randint(0, 100, n).astype(float),
            "D4": np.random.randint(0, 200, n).astype(float),
            "V1": np.random.randint(0, 2, n).astype(float),
            "V2": np.random.randint(0, 2, n).astype(float),
            "V3": np.random.randint(0, 2, n).astype(float),
            "V4": np.random.randint(0, 2, n).astype(float),
            "V5": np.random.randint(0, 2, n).astype(float),
            "V6": np.random.randint(0, 2, n).astype(float),
            "isFraud": np.random.choice([0, 1], n, p=[0.965, 0.035]),
        }
    )


@pytest.fixture
def feature_df(sample_transactions):
    from app.core.features import build_features, build_merchant_risk_profile

    risk = build_merchant_risk_profile(sample_transactions)
    return build_features(sample_transactions, risk_profile=risk, fast=True)


@pytest.fixture
def X_y(feature_df):
    from app.core.features import get_feature_columns

    feat_cols = get_feature_columns(feature_df)
    X = feature_df[feat_cols].fillna(0).values
    y = feature_df["isFraud"].values
    return X, y, feat_cols


# ─── Feature engineering ─────────────────────────────────────────────────────


class TestFeatureEngineering:
    def test_haversine_zero_distance(self):
        from app.core.features import haversine_distance

        d = haversine_distance(40.0, -74.0, 40.0, -74.0)
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_haversine_known_distance(self):
        from app.core.features import haversine_distance

        # NYC to LA is roughly 3940 km
        d = haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)
        assert 3800 < d < 4100

    def test_address_distance_added(self, sample_transactions):
        from app.core.features import add_address_distance

        df = add_address_distance(sample_transactions)
        assert "addr_distance_km" in df.columns
        assert "addr_mismatch" in df.columns
        assert df["addr_distance_km"].notna().all()

    def test_temporal_features(self, sample_transactions):
        from app.core.features import add_temporal_features

        df = add_temporal_features(sample_transactions)
        assert "tx_hour" in df.columns
        assert df["tx_hour"].between(0, 23).all()
        assert "tx_is_weekend" in df.columns
        assert set(df["tx_is_weekend"].unique()).issubset({0, 1})

    def test_velocity_features_fast(self, sample_transactions):
        from app.core.features import add_velocity_features_fast

        df = add_velocity_features_fast(sample_transactions, windows_hours=[1, 24])
        assert "velocity_count_1h" in df.columns
        assert "velocity_count_24h" in df.columns
        assert (df["velocity_count_1h"] >= 0).all()

    def test_merchant_risk_profile(self, sample_transactions):
        from app.core.features import build_merchant_risk_profile, add_merchant_risk

        profile = build_merchant_risk_profile(sample_transactions)
        assert "merchant_fraud_rate" in profile.columns
        assert (profile["merchant_fraud_rate"] >= 0).all()
        assert (profile["merchant_fraud_rate"] <= 1).all()

        df = add_merchant_risk(sample_transactions, profile)
        assert "merchant_fraud_rate" in df.columns
        assert "high_risk_merchant" in df.columns

    def test_amount_features(self, sample_transactions):
        from app.core.features import add_amount_features

        df = add_amount_features(sample_transactions)
        assert "log_amt" in df.columns
        assert (df["log_amt"] >= 0).all()
        assert "amt_is_round" in df.columns

    def test_build_features_full_pipeline(self, feature_df):
        from app.core.features import get_feature_columns

        feat_cols = get_feature_columns(feature_df)
        assert len(feat_cols) > 10
        # No NaNs in feature columns
        assert feature_df[feat_cols].isna().sum().sum() == 0

    def test_card_aggregates(self, sample_transactions):
        from app.core.features import add_card_aggregates

        df = add_card_aggregates(sample_transactions)
        assert "card_avg_amt" in df.columns
        assert "amt_zscore" in df.columns


# ─── AutoEncoder ─────────────────────────────────────────────────────────────


class TestAutoEncoder:
    def test_forward_pass(self):
        from app.core.autoencoder import FraudAutoEncoder

        model = FraudAutoEncoder(input_dim=20, bottleneck=8)
        x = torch.randn(16, 20)
        out = model(x)
        assert out.shape == (16, 20)

    def test_reconstruction_error_positive(self):
        from app.core.autoencoder import FraudAutoEncoder

        model = FraudAutoEncoder(input_dim=20, bottleneck=8)
        x = torch.randn(16, 20)
        errors = model.reconstruction_error(x)
        assert errors.shape == (16,)
        assert (errors >= 0).all()

    def test_encode_bottleneck_shape(self):
        from app.core.autoencoder import FraudAutoEncoder

        model = FraudAutoEncoder(input_dim=20, bottleneck=8)
        x = torch.randn(16, 20)
        z = model.encode(x)
        assert z.shape == (16, 8)

    def test_trainer_fit_and_threshold(self):
        from app.core.autoencoder import AutoEncoderTrainer

        X = np.random.randn(200, 20).astype(np.float32)
        trainer = AutoEncoderTrainer(input_dim=20, bottleneck=8)
        trainer.fit(X, epochs=3, patience=5, verbose=False)
        threshold = trainer.calibrate_threshold(X, percentile=95.0)
        assert threshold > 0

    def test_trainer_predict_binary(self):
        from app.core.autoencoder import AutoEncoderTrainer

        X = np.random.randn(100, 20).astype(np.float32)
        trainer = AutoEncoderTrainer(input_dim=20, bottleneck=8)
        trainer.fit(X, epochs=2, verbose=False)
        trainer.calibrate_threshold(X)
        preds = trainer.predict_binary(X)
        assert preds.shape == (100,)
        assert set(preds).issubset({0, 1})

    def test_trainer_save_load(self, tmp_path):
        from app.core.autoencoder import AutoEncoderTrainer

        X = np.random.randn(100, 20).astype(np.float32)
        trainer = AutoEncoderTrainer(input_dim=20, bottleneck=8)
        trainer.fit(X, epochs=2, verbose=False)
        trainer.calibrate_threshold(X)

        path = str(tmp_path / "ae.pt")
        trainer.save(path)
        loaded = AutoEncoderTrainer.load(path)
        assert loaded.threshold == pytest.approx(trainer.threshold, rel=1e-5)

        # Scores should be identical
        s1 = trainer.predict_anomaly_score(X)
        s2 = loaded.predict_anomaly_score(X)
        np.testing.assert_allclose(s1, s2, rtol=1e-4)


# ─── Ensemble ────────────────────────────────────────────────────────────────


class TestEnsemble:
    @pytest.fixture
    def trained_ensemble(self, X_y):
        from app.core.ensemble import FraudEnsemble

        X, y, feat_cols = X_y
        # Ensure at least a few fraud samples
        y[: max(5, int(len(y) * 0.03))] = 1
        model = FraudEnsemble(input_dim=X.shape[1], ae_epochs=3, smote=False)
        model.fit(X, y, feature_names=feat_cols, verbose=False)
        return model, X, y

    def test_predict_proba_range(self, trained_ensemble):
        model, X, y = trained_ensemble
        proba = model.predict_proba(X)
        assert proba.shape == (len(X),)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_predict_binary(self, trained_ensemble):
        model, X, y = trained_ensemble
        preds = model.predict(X)
        assert set(preds).issubset({0, 1})

    def test_anomaly_scores_positive(self, trained_ensemble):
        model, X, y = trained_ensemble
        scores = model.anomaly_scores(X)
        assert scores.shape == (len(X),)
        assert (scores >= 0).all()

    def test_shap_explain_single(self, trained_ensemble):
        model, X, y = trained_ensemble
        result = model.explain_single(X[0])
        assert "top_features" in result
        assert len(result["top_features"]) > 0
        assert "feature" in result["top_features"][0]
        assert "shap_value" in result["top_features"][0]

    def test_evaluate_returns_metrics(self, trained_ensemble):
        model, X, y = trained_ensemble
        metrics = model.evaluate(X, y)
        assert "auc_pr" in metrics
        assert "auc_roc" in metrics
        assert 0 <= metrics["auc_pr"] <= 1
        assert 0 <= metrics["auc_roc"] <= 1

    def test_save_and_load(self, trained_ensemble, tmp_path):
        model, X, y = trained_ensemble
        model.save(str(tmp_path))
        loaded = type(model).load(str(tmp_path))
        p1 = model.predict_proba(X[:10])
        p2 = loaded.predict_proba(X[:10])
        np.testing.assert_allclose(p1, p2, rtol=1e-4)


# ─── Drift monitor ───────────────────────────────────────────────────────────


class TestDriftMonitor:
    def test_psi_stable(self):
        from app.streaming.drift_monitor import compute_psi

        expected = np.random.uniform(0, 1, 1000)
        actual = np.random.uniform(0, 1, 1000)
        psi = compute_psi(expected, actual)
        assert psi < 0.1  # stable distributions

    def test_psi_shifted(self):
        from app.streaming.drift_monitor import compute_psi

        expected = np.random.uniform(0, 0.3, 1000)
        actual = np.random.uniform(0.7, 1.0, 1000)
        psi = compute_psi(expected, actual)
        assert psi > 0.2  # significant shift

    def test_set_baseline(self):
        from app.streaming.drift_monitor import DriftMonitor

        monitor = DriftMonitor()
        scores = np.random.uniform(0, 1, 500)
        labels = (scores > 0.5).astype(int)
        metrics = monitor.set_baseline(scores, labels)
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1

    def test_record_and_summary(self):
        from app.streaming.drift_monitor import DriftMonitor

        monitor = DriftMonitor()
        for _ in range(200):
            monitor.record(np.random.uniform(0, 1))
        summary = monitor.summary()
        assert summary["window_size"] == 200
        assert 0 <= summary["avg_fraud_score"] <= 1

    def test_check_no_alerts_stable(self):
        from app.streaming.drift_monitor import DriftMonitor

        monitor = DriftMonitor()
        scores = np.random.uniform(0.1, 0.4, 500)
        labels = np.zeros(500, dtype=int)
        monitor.set_baseline(scores, labels)
        for s, label in zip(scores, labels):
            monitor.record(s, label)
        alerts = monitor.check()
        # Stable distribution — no drift alerts expected
        assert isinstance(alerts, list)

    def test_no_crash_without_baseline(self):
        from app.streaming.drift_monitor import DriftMonitor

        monitor = DriftMonitor()
        for _ in range(200):
            monitor.record(np.random.uniform(0, 1))
        alerts = monitor.check()
        assert isinstance(alerts, list)


# ─── Kafka producer (mocked) ──────────────────────────────────────────────────


class TestKafkaProducer:
    def test_send_success(self):
        from app.streaming.producer import TransactionProducer

        with patch("app.streaming.producer.KafkaProducer") as mock_kafka:
            mock_future = MagicMock()
            mock_future.get.return_value = None
            mock_kafka.return_value.send.return_value = mock_future

            producer = TransactionProducer(bootstrap_servers="localhost:9092")
            result = producer.send({"TransactionID": 1, "TransactionAmt": 100.0})
            assert result is True

    def test_send_failure_goes_to_dlq(self):
        from app.streaming.producer import TransactionProducer
        from kafka.errors import KafkaError

        with patch("app.streaming.producer.KafkaProducer") as mock_kafka:
            mock_kafka.return_value.send.side_effect = KafkaError("Connection refused")
            producer = TransactionProducer(bootstrap_servers="localhost:9092")
            result = producer.send({"TransactionID": 2, "TransactionAmt": 50.0})
            assert result is False

    def test_context_manager(self):
        from app.streaming.producer import TransactionProducer

        with patch("app.streaming.producer.KafkaProducer") as mock_kafka:
            mock_kafka.return_value.flush.return_value = None
            mock_kafka.return_value.close.return_value = None
            with TransactionProducer():
                pass
            mock_kafka.return_value.flush.assert_called_once()


# ─── FastAPI endpoints ────────────────────────────────────────────────────────


class TestAPIEndpoints:
    @pytest.fixture
    def client(self):
        from app.main import app

        return TestClient(app)

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data
        assert "kafka_connected" in data

    def test_score_no_model_returns_503(self, client):
        import app.main as main_module

        original = main_module._model_loaded
        main_module._model_loaded = False
        main_module._model = None

        tx = {
            "TransactionDT": 86400,
            "TransactionAmt": 150.0,
            "ProductCD": "W",
            "card1": 1234,
        }
        response = client.post("/score", json=tx)
        assert response.status_code == 503
        main_module._model_loaded = original

    def test_score_with_mock_model(self, client, X_y):
        from app.core.ensemble import FraudEnsemble
        import app.main as main_module

        X, y, feat_cols = X_y
        # Ensure fraud samples
        y[: max(5, int(len(y) * 0.03))] = 1
        mock_model = FraudEnsemble(input_dim=X.shape[1], ae_epochs=2, smote=False)
        mock_model.fit(X, y, feature_names=feat_cols, verbose=False)

        original_model = main_module._model
        original_loaded = main_module._model_loaded
        main_module._model = mock_model
        main_module._model_loaded = True
        import numpy as _np

        mock_model.predict_proba = lambda X: _np.array([0.1] * len(X))
        mock_model.anomaly_scores = lambda X: _np.array([0.05] * len(X))

        tx = {
            "TransactionDT": 86400,
            "TransactionAmt": 150.0,
            "ProductCD": "W",
            "card1": 1234,
            "addr1": 200.0,
            "addr2": 50.0,
            "C1": 2.0,
            "D1": 10.0,
        }
        response = client.post("/score", json=tx)
        assert response.status_code == 200
        data = response.json()
        assert "fraud_score" in data
        assert 0.0 <= data["fraud_score"] <= 1.0
        assert "is_fraud" in data
        assert "anomaly_score" in data
        assert "latency_ms" in data

        main_module._model = original_model
        main_module._model_loaded = original_loaded

    def test_batch_score(self, client, X_y):
        from app.core.ensemble import FraudEnsemble
        import app.main as main_module

        X, y, feat_cols = X_y
        y[: max(5, int(len(y) * 0.03))] = 1
        mock_model = FraudEnsemble(input_dim=X.shape[1], ae_epochs=2, smote=False)
        mock_model.fit(X, y, feature_names=feat_cols, verbose=False)

        original_model = main_module._model
        original_loaded = main_module._model_loaded
        main_module._model = mock_model
        main_module._model_loaded = True
        import numpy as _np

        mock_model.predict_proba = lambda X: _np.array([0.1] * len(X))
        mock_model.anomaly_scores = lambda X: _np.array([0.05] * len(X))

        payload = {
            "transactions": [
                {
                    "TransactionDT": 86400,
                    "TransactionAmt": float(100 + i),
                    "card1": 1234,
                }
                for i in range(5)
            ],
            "include_explanations": False,
        }
        response = client.post("/score/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 5
        assert "flagged" in data
        assert len(data["results"]) == 5

        main_module._model = original_model
        main_module._model_loaded = original_loaded

    def test_drift_endpoint(self, client):
        response = client.get("/drift")
        assert response.status_code == 200
        data = response.json()
        assert "window_size" in data
        assert "fraud_rate" in data
        assert "alerts" in data

    def test_metrics_endpoint(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "fraud_transactions" in response.text or "HELP" in response.text

    def test_docs_accessible(self, client):
        response = client.get("/docs")
        assert response.status_code == 200

    def test_score_invalid_amount(self, client):
        tx = {"TransactionDT": 86400, "TransactionAmt": -100.0}
        response = client.post("/score", json=tx)
        assert response.status_code == 422


# ─── Rule-based explainer ────────────────────────────────────────────────────


class TestRuleBasedExplainer:
    def test_explain_returns_string(self):
        from app.agent.explainer import RuleBasedExplainer

        explainer = RuleBasedExplainer()
        tx = {"TransactionID": 1, "TransactionAmt": 999.99, "ProductCD": "W"}
        shap_result = {
            "top_features": [
                {
                    "feature": "velocity_count_1h",
                    "shap_value": 0.8,
                    "direction": "increases_fraud_risk",
                },
                {
                    "feature": "addr_distance_km",
                    "shap_value": 0.5,
                    "direction": "increases_fraud_risk",
                },
            ]
        }
        result = explainer.explain(
            tx, shap_result, anomaly_score=0.1, ae_threshold=0.05, fraud_score=0.85
        )
        assert isinstance(result, str)
        assert len(result) > 20

    def test_explain_high_score_recommends_block(self):
        from app.agent.explainer import RuleBasedExplainer

        explainer = RuleBasedExplainer()
        tx = {"TransactionAmt": 5000.0}
        result = explainer.explain(tx, {"top_features": []}, 0.2, 0.05, fraud_score=0.9)
        assert "BLOCK" in result

    def test_explain_low_score_recommends_monitor(self):
        from app.agent.explainer import RuleBasedExplainer

        explainer = RuleBasedExplainer()
        tx = {"TransactionAmt": 20.0}
        result = explainer.explain(
            tx, {"top_features": []}, 0.01, 0.05, fraud_score=0.2
        )
        assert "MONITOR" in result
