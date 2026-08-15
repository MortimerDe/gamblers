import numpy as np

from gamblers.core.machines.coin import CoinFlipper, CoinFlipperConfig
from gamblers.core.types import AgentId, MachineId


def main() -> None:
    config = CoinFlipperConfig(
        machine_id=MachineId("coin_flipper_1"),
        pos=(0, 0),
        cap=5,
        eps=0.05,
        delay_ticks=2,
    )
    machine = CoinFlipper(config=config)

    rng = np.random.default_rng(seed=42)
    capital = 0
    total_delta = 0
    plays = 10_000_000

    for _ in range(plays):
        outcome = machine.play(AgentId(1), capital, rng)
        capital += outcome.delta
        total_delta += outcome.delta

    m = total_delta / plays
    print(f"Final Capital after {plays} plays: {capital}")
    print(f"M({plays} plays): {m}")

if __name__ == "__main__":
    main()