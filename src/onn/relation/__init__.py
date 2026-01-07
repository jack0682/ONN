"""ONN Relation Encoding Package.

Provides pair feature extraction and relation embedding parameterization.
Now includes SE(3) based label-free relation encoding.

Reference:
    spec/20_impl_plan.ir.yml: IMPL_019
"""

from onn.relation.basis import (
    compute_pair_features,
    compute_pair_features_batch,
    standardize_features,
)

from onn.relation.param import (
    RelationParamConfig,
    RelationEncoder,
    encode_relation,
    random_projection,
    create_default_encoder,
)

from onn.relation.se3_encoder import (
    SE3RelationEncoder,
    GeometricRelationEncoder,
    log_SO3,
    exp_SO3,
    log_SE3,
    exp_SE3,
    relative_transform,
)

from onn.relation.predictive_relation import (
    PredictiveRelationModel,
    RelationPredictionConfig,
    RelationDataGenerator,
)

from onn.relation.four_axis_relation import (
    FourAxisRelationModel,
    RelationAxisConfig,
    OutcomeSignature,
)

from onn.relation.contrastive_relation import (
    ContrastiveRelationLearner,
    ContrastiveConfig,
    compute_outcome_signature,
    signature_similarity,
)

__all__ = [
    # Basis
    "compute_pair_features",
    "compute_pair_features_batch",
    "standardize_features",
    # Param
    "RelationParamConfig",
    "RelationEncoder",
    "encode_relation",
    "random_projection",
    "create_default_encoder",
    # SE3
    "SE3RelationEncoder",
    "GeometricRelationEncoder",
    "log_SO3",
    "exp_SO3",
    "log_SE3",
    "exp_SE3",
    "relative_transform",
    # Predictive
    "PredictiveRelationModel",
    "RelationPredictionConfig",
    "RelationDataGenerator",
    # 4-Axis
    "FourAxisRelationModel",
    "RelationAxisConfig",
    "OutcomeSignature",
    # Contrastive (NEW)
    "ContrastiveRelationLearner",
    "ContrastiveConfig",
    "compute_outcome_signature",
    "signature_similarity",
]

