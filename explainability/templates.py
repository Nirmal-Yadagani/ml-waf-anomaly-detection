class ExplanationTemplates:
    TEMPLATES = {
        "req_rate": "Request rate is {value:.1f}/min, exceeding baseline (p99={p99:.1f})",
        "unique_uri_count": "Multiple endpoints accessed unusually fast",
        "payload_size_mean": "Average payload size ({value:.0f} bytes) is unusually large",
        "payload_entropy": "High-entropy payload detected (entropy={value:.2f})",
        "error_rate_4xx": "High rate of client errors (4xx responses)",
        "error_rate_5xx": "Server error rate elevated (5xx responses)",
        "avg_response_time": "Response time degraded (avg={value:.1f} ms)",
        "endpoint_rarity": "Rare endpoint accessed",
        "interarrival_std": "Non-human timing pattern detected",
        "burstiness": "Traffic burst behavior detected",
    }

    @classmethod
    def render(cls, feature: str, value, p99) -> str:
        template = cls.TEMPLATES.get(feature)
        if not template:
            return None

        # --- Force scalar conversion ---
        if hasattr(value, "iloc"):
            value = value.iloc[0]
        if hasattr(p99, "iloc"):
            p99 = p99.iloc[0]

        value = float(value)
        p99 = float(p99)

        return template.format(value=value, p99=p99)

