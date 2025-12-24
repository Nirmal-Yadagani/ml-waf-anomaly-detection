import pickle
from pathlib import Path

from ingestion.log_reader import NginxLogReader
from feature_engineering.extractor import FeatureExtractor
from baseline.baseline_trainer import BaselineTrainer
from anomaly_detection.isolation_forest import IsolationForestModel
from anomaly_detection.scorer import AnomalyScorer
from explainability.feature_diff import FeatureDiff
from explainability.explanation_builder import ExplanationBuilder
from rule_engine.rule_generator import RuleGenerator
from rule_engine.rule_validator import RuleValidator
from baseline.baseline_store import BaselineStore
from training.retraining_hooks import RetrainingHooks

# --------------------------------------------------
# 1. Read traffic logs
# --------------------------------------------------

log_path = "/home/nirmal-yadagani/synthetic_logs/ml_access.log"
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

# -------------------------
# Adaptive baseline
# -------------------------

baseline_store = BaselineStore()
baseline_trainer = BaselineTrainer()

existing_baseline = baseline_store.load()
if existing_baseline:
    baseline_trainer.baseline = existing_baseline
    print("[INFO] Loaded existing baseline")
else:
    print("[INFO] No existing baseline found")

# Initial scoring (before update)
baseline_scores = baseline_trainer.score_deviation(ml_features) \
    if baseline_trainer.baseline else ml_features.iloc[:, 0] * 0

# -------------------------
# Retraining decision
# -------------------------

hooks = RetrainingHooks()
avg_baseline_score = baseline_scores.mean()

should_update = hooks.should_retrain(
    fp_rate=0.0,   # hook for admin feedback later
    avg_baseline_score=avg_baseline_score,
)

if should_update:
    if baseline_trainer.baseline:
        baseline_trainer.update(ml_features, alpha=0.1)
        print("[INFO] Baseline updated adaptively")
    else:
        baseline_trainer.fit(ml_features)
        print("[INFO] Baseline trained (cold start)")

    baseline_store.save(baseline_trainer.get_baseline())
    hooks.mark_retrained()
else:
    print("[INFO] Baseline remains unchanged")

baseline = baseline_trainer.get_baseline()
baseline_scores = baseline_trainer.score_deviation(ml_features)


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
