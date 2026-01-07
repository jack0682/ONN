"""ONN ES Training Package.

Provides CMA-ES optimization for ONN solver hyperparameters.

Reference:
    spec/20_impl_plan.ir.yml: IMPL_020, IMPL_021
"""

from onn.es.ask_tell import (
    ESConfig,
    CMAESState,
    CMAESTrainer,
    Candidate,
    create_default_trainer,
    create_trainer_for_params,
    create_trainer_with_w_lin,
)

from onn.es.fitness import (
    FitnessConfig,
    EpisodeStep,
    evaluate_episode,
    evaluate_candidate,
    aggregate_metrics,
    compute_fitness,
    generate_synthetic_episode,
)

from onn.es.schedule import (
    ScheduleConfig,
    linear_schedule,
    exponential_schedule,
    cosine_schedule,
    apply_schedule,
    create_warmup_schedule,
)

from onn.es.geometric_integration import (
    # Configuration
    GeometricFitnessConfig,
    GeometricEpisodeStep,
    GeometricEvalMetrics,
    # Encoder
    GeometricRelationEncoderES,
    # Evaluation
    evaluate_geometric_episode,
    compute_geometric_fitness,
    evaluate_geometric_candidate,
    # Generation
    generate_geometric_episode,
    # Factory
    create_geometric_encoder_es,
    create_geometric_fitness_config,
)

from onn.es.hybrid_trainer import (
    # Config
    HybridTrainerConfig,
    SurrogateLossConfig,
    # Classes
    HybridTrainer,
    LearnableEncoder,
    SurrogateLoss,
    # Factory
    create_hybrid_trainer,
    create_surrogate_loss,
)

__all__ = [
    # Ask/Tell
    "ESConfig",
    "CMAESState",
    "CMAESTrainer",
    "Candidate",
    "create_default_trainer",
    "create_trainer_for_params",
    "create_trainer_with_w_lin",
    # Fitness
    "FitnessConfig",
    "EpisodeStep",
    "evaluate_episode",
    "evaluate_candidate",
    "aggregate_metrics",
    "compute_fitness",
    "generate_synthetic_episode",
    # Schedule
    "ScheduleConfig",
    "linear_schedule",
    "exponential_schedule",
    "cosine_schedule",
    "apply_schedule",
    "create_warmup_schedule",
    # Geometric Integration
    "GeometricFitnessConfig",
    "GeometricEpisodeStep",
    "GeometricEvalMetrics",
    "GeometricRelationEncoderES",
    "evaluate_geometric_episode",
    "compute_geometric_fitness",
    "evaluate_geometric_candidate",
    "generate_geometric_episode",
    "create_geometric_encoder_es",
    "create_geometric_fitness_config",
    # Hybrid ES + GD
    "HybridTrainerConfig",
    "SurrogateLossConfig",
    "HybridTrainer",
    "LearnableEncoder",
    "SurrogateLoss",
    "create_hybrid_trainer",
    "create_surrogate_loss",
]
