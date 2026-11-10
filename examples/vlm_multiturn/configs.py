from dataclasses import dataclass, field

from areal.api.cli_args import GRPOConfig


@dataclass
class VisionMultiTurnGRPOConfig(GRPOConfig):
    """Config for multi-turn VLM GRPO training."""

    max_turns: int = field(
        default=2,
        metadata={
            "help": "Maximum number of turns for multi-turn agentic interaction."
        },
    )
    turn_discount: float = field(
        default=0.95,
        metadata={
            "help": "Discount factor for rewards at each turn. Used to incentivize "
            "faster correct answers."
        },
    )
