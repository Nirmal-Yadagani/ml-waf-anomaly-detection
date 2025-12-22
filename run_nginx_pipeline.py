from ingestion.log_reader import NginxLogReader
from feature_engineering.extractor import FeatureExtractor
from baseline.baseline_trainer import BaselineTrainer
from anomaly_detection.isolation_forest import IsolationForestModel
from anomaly_detection.scorer import AnomalyScorer
from explainability.feature_diff import FeatureDiff
from explainability.explanation_builder import ExplanationBuilder
from rule_engine.rule_generator import RuleGenerator
from rule_engine.rule_validator import RuleValidator

# --------------------------------------------------
# 1. Read traffic logs
# --------------------------------------------------

log_path = "/var/log/nginx/ml_access.log"
events = list(NginxLogReader(log_path).read())

print(f"[INFO] Loaded {len(events)} traffic events")

if len(events) < 50:
    raise RuntimeError("Not enough traffic yet. Generate some requests.")

# --------------------------------------------------
# 2. Feature extraction
# --------------------------------------------------

extractor = FeatureExtractor(window="1min")
output = extractor.extract(events)

context = output["context"]
behavioral = output["behavioral_features"]
ml_features = output["ml_features"]

print("[INFO] Feature extraction complete")

# --------------------------------------------------
# 3. Baseline learning
# --------------------------------------------------

baseline_trainer = BaselineTrainer()
baseline_trainer.fit(ml_features)

baseline_scores = baseline_trainer.score_deviation(ml_features)

print("[INFO] Baseline learned")

# --------------------------------------------------
# 4. Isolation Forest
# --------------------------------------------------

if_model = IsolationForestModel(contamination=0.02)
if_model.fit(ml_features)

if_scores = if_model.score(ml_features)

print("[INFO] Isolation Forest trained")

# --------------------------------------------------
# 5. Hybrid anomaly scoring
# --------------------------------------------------

scorer = AnomalyScorer(
    if_weight=0.6,
    baseline_weight=0.4,
    anomaly_threshold=0.75
)

results = scorer.score(if_scores, baseline_scores)

print("[INFO] Anomaly scoring complete")

# --------------------------------------------------
# 6. Explainability
# --------------------------------------------------

baseline = baseline_trainer.get_baseline()

diff_engine = FeatureDiff(baseline)
diffs = diff_engine.diff(ml_features)

explainer = ExplanationBuilder(baseline)
explanations = explainer.build(ml_features, diffs)

results["explanations"] = results.index.map(
    lambda i: explanations.get(i, [])
)

print("[INFO] Explanations generated")

# --------------------------------------------------
# 7. Rule generation
# --------------------------------------------------

rule_gen = RuleGenerator(
    template_path="rule_engine/rule_templates.yaml",
    min_occurrences=1,
    min_avg_score=0.75
)

raw_rules = rule_gen.generate(
    context=context,
    results=results
)

validator = RuleValidator(
    protected_endpoints=["/health"]
)

rules = [r for r in raw_rules if validator.validate(r)]

# --------------------------------------------------
# 8. Print results
# --------------------------------------------------

print("\n=== DETECTED ANOMALIES ===")
anomalies = results[results["is_anomaly"]]

for idx, row in anomalies.iterrows():
    print(f"\nScore: {row['final_score']:.2f}")
    for reason in row["explanations"]:
        print(" -", reason)

print("\n=== PROPOSED RULES ===")
for rule in rules:
    print(rule)


import pickle
from pathlib import Path

Path("dashboard/data").mkdir(parents=True, exist_ok=True)

with open("dashboard/data/context.pkl", "wb") as f:
    pickle.dump(context, f)

with open("dashboard/data/ml_features.pkl", "wb") as f:
    pickle.dump(ml_features, f)

with open("dashboard/data/results.pkl", "wb") as f:
    pickle.dump(results, f)

with open("dashboard/data/rules.pkl", "wb") as f:
    pickle.dump(rules, f)

with open("dashboard/data/baseline.pkl", "wb") as f:
    pickle.dump(baseline, f)
