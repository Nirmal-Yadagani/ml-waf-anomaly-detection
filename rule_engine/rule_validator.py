# rule_engine/rule_validator.py

class RuleValidator:
    def __init__(self, protected_endpoints=None):
        self.protected_endpoints = protected_endpoints or []

    def validate(self, rule: dict) -> bool:
        """
        Returns True if rule is safe to propose
        """

        endpoint = rule["match"].get("endpoint")

        if endpoint in self.protected_endpoints:
            return False

        if rule["confidence"] < 0.8:
            return False

        return True
