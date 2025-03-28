import os
import numpy as np
import json
import socket
import platform
import subprocess
from datetime import datetime

class IdentityManager:
    def __init__(self, snapshot_dir):
        self.snapshot_dir = snapshot_dir
        self.meta_path = os.path.join(snapshot_dir, "self_meta.json")
        self.memory_path = os.path.join(snapshot_dir, "self_memory.npy")
        self.self_history = []
        self.meta_info = {}

        self._load_or_init()

    def _load_or_init(self):
        if os.path.exists(self.meta_path):
            with open(self.meta_path, 'r') as f:
                self.meta_info = json.load(f)
        else:
            self.meta_info = {
                "identity_id": self._generate_id(),
                "created_at": str(datetime.now()),
                "updates": 0,
                "history": []
            }

        if os.path.exists(self.memory_path):
            self.self_history = list(np.load(self.memory_path, allow_pickle=True))
        else:
            self.self_history = []

    def _generate_id(self):
        return f"SELF-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    def update_self(self, new_self_vector, context_label):
        self_description = self._self_inquiry()
        abstraction_features = self._vectorize_self_description(self_description)
        fused_self = np.concatenate([new_self_vector, abstraction_features])

        entry = {
            "timestamp": str(datetime.now()),
            "label": context_label,
            "vector": fused_self.tolist(),
            "self_description": self_description
        }
        self.self_history.append(entry)
        self.meta_info["updates"] += 1
        self.meta_info["history"].append({"timestamp": entry["timestamp"], "label": context_label})

        # Save memory and metadata
        np.save(self.memory_path, np.array(self.self_history, dtype=object))
        with open(self.meta_path, 'w') as f:
            json.dump(self.meta_info, f, indent=2)

    def _self_inquiry(self):
        try:
            hostname = socket.gethostname()
            ip_address = socket.gethostbyname(hostname)
            system_info = platform.uname()
            proxies = os.environ.get("http_proxy") or os.environ.get("https_proxy")
            services = self._scan_local_ports()
            return {
                "hostname": hostname,
                "ip": ip_address,
                "os": system_info.system,
                "node": system_info.node,
                "release": system_info.release,
                "version": system_info.version,
                "machine": system_info.machine,
                "processor": system_info.processor,
                "proxy": proxies,
                "services": services
            }
        except Exception as e:
            return {"error": str(e)}

    def _scan_local_ports(self):
        try:
            result = subprocess.run(["ss", "-tuln"], capture_output=True, text=True)
            return result.stdout.splitlines()
        except Exception as e:
            return [f"Failed to scan ports: {e}"]

    def _vectorize_self_description(self, desc):
        # Simple feature encoding for now — convert known fields to numeric representation
        vector = []
        for key in ["os", "release", "version", "machine", "processor"]:
            val = desc.get(key, "unknown")
            vector.append(hash(val) % 10000 / 10000.0)  # Normalize
        vector.append(1.0 if desc.get("proxy") else 0.0)
        vector.append(len(desc.get("services", [])) / 100.0)  # Normalize service count
        return np.array(vector, dtype=np.float32)

    def get_last_self(self):
        if self.self_history:
            return np.array(self.self_history[-1]["vector"])
        else:
            return None

    def get_meta(self):
        return self.meta_info

    def get_all_self_vectors(self):
        return [np.array(e["vector"]) for e in self.self_history]

    def get_labels(self):
        return [e["label"] for e in self.self_history]
