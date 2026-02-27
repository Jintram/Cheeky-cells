
# %% ################################################################################
# Originally written by Martijn Wehrens,
# refactored by ChatGPT Codex 5.3 to streamline preliminary code.


# %% ################################################################################
# Arabidopsis ML pipeline orchestrator
#
# This wrapper calls three isolated phase scripts:
# 1) create training set
# 2) train model
# 3) apply model
#
# Each phase has its own config dataclass, so phases are independent.

from dataclasses import dataclass

from pipeline_example_arabidopsis_ML_phase1_create_training_set import Phase1Config, run_phase1_pipeline
from pipeline_example_arabidopsis_ML_phase2_train_model import Phase2Config, run_phase2_pipeline
from pipeline_example_arabidopsis_ML_phase3_apply_model import Phase3Config, run_phase3_pipeline


# %% ################################################################################
# Orchestrator settings

@dataclass
class OrchestratorConfig:
    # Toggle each phase independently
    run_phase1: bool = True
    run_phase2: bool = True
    run_phase3: bool = False

    # Optional checkpoint override for phase 3
    # If left as None, phase3 will use its own default in Phase3Config
    phase3_checkpoint_override: str | None = None


# %% ################################################################################
# Orchestrator runner


def run_default_pipeline(config: OrchestratorConfig) -> None:
    # ---------------------------------------------------------------------
    # Part 1: create training set
    # ---------------------------------------------------------------------
    if config.run_phase1:
        print('\n--- Part 1: creating training set ---')
        phase1_config = Phase1Config()
        run_phase1_pipeline(phase1_config)

    # ---------------------------------------------------------------------
    # Part 2: train model
    # ---------------------------------------------------------------------
    phase2_saved_checkpoint = None
    if config.run_phase2:
        print('\n--- Part 2: training model ---')
        phase2_config = Phase2Config()
        phase2_saved_checkpoint = run_phase2_pipeline(phase2_config)

    # ---------------------------------------------------------------------
    # Part 3: apply model
    # ---------------------------------------------------------------------
    if config.run_phase3:
        print('\n--- Part 3: applying model ---')
        phase3_config = Phase3Config()

        # Priority for checkpoint path:
        # 1) explicit orchestrator override
        # 2) checkpoint returned by phase 2 in this run
        # 3) default value in Phase3Config
        if config.phase3_checkpoint_override is not None:
            phase3_config.model_checkpoint_to_load = config.phase3_checkpoint_override
        elif phase2_saved_checkpoint is not None:
            phase3_config.model_checkpoint_to_load = phase2_saved_checkpoint

        run_phase3_pipeline(phase3_config)


# %% ################################################################################
# Execute orchestrator directly

if __name__ == '__main__':
    default_config = OrchestratorConfig()
    run_default_pipeline(default_config)
