"""Exercise effects and startup inside an x86 CPU-constrained container."""

import tempfile

import numpy as np
from fastapi.testclient import TestClient

from backend import config
from backend.app import app
from backend.database.models import EffectPreset
from backend.utils.effects import apply_effects


def main() -> None:
    signal = np.full(1024, 0.1, dtype=np.float32)
    processed = apply_effects(
        signal,
        16000,
        [{"type": "gain", "enabled": True, "params": {"gain_db": 6.0}}],
    )
    np.testing.assert_allclose(processed, signal * (10 ** (6.0 / 20)), rtol=1e-4)

    with tempfile.TemporaryDirectory(prefix="voicebox-ivy-") as data_dir:
        config.set_data_dir(data_dir)
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200, response.text

            from backend.database import session

            with session.SessionLocal() as db:
                assert db.query(EffectPreset).count() > 0

    print("Effects, database seeding, and /health passed")


if __name__ == "__main__":
    main()
