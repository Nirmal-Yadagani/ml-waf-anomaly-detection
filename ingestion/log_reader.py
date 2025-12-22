import json
from ingestion.schema import TrafficEvent
from datetime import datetime

class NginxLogReader:
    def __init__(self, path):
        self.path = path

    def read(self):
        with open(self.path) as f:
            for line in f:
                data = json.loads(line)
                yield TrafficEvent(
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    src_ip=data["src_ip"],
                    method=data["method"],
                    uri_path=data["uri_path"],
                    status_code=data["status_code"],
                    payload_size=data["payload_size"],
                    response_time_ms=float(data["response_time_ms"]) * 1000,
                    user_agent=data["user_agent"],
                )
