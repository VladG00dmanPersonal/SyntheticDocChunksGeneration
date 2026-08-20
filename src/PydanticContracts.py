from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------
# Synthetic examples
# ---------------------------------------------------------------------


class ChunkingVariant(StrictBaseModel):
    chunks: list[str] = Field(min_length=1)
    rationale: str = Field(min_length=1)
    focus: dict[str, Any] = Field(default_factory=dict)


class SyntheticChunkingExample(StrictBaseModel):
    document_title: str = Field(min_length=1)
    source_document: str = Field(min_length=1)

    positive: ChunkingVariant
    negative: ChunkingVariant

    controlled_change: str = Field(min_length=1)

    expected_relation: Literal["positive_higher_than_negative"]


# ---------------------------------------------------------------------
# Common judge contracts
# ---------------------------------------------------------------------


JudgeIssueSeverity = Literal["fatal", "major", "minor"]


class JudgeIssue(StrictBaseModel):
    severity: JudgeIssueSeverity
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)


ChecksT = TypeVar("ChecksT", bound=BaseModel)


class JudgeResult(StrictBaseModel, Generic[ChecksT]):
    valid: bool
    quality_score: int = Field(ge=1, le=5)

    issues: list[JudgeIssue] = Field(default_factory=list)
    checks: ChecksT

    reason: str = Field(min_length=1)


# ---------------------------------------------------------------------
# Size Compliance
# ---------------------------------------------------------------------


class SizeComplianceChecks(StrictBaseModel):
    same_source_text: bool
    boundary_only_change: bool
    positive_size_compliant: bool
    negative_has_size_violation: bool
    focus_valid: bool
    controlled_change_valid: bool
    metric_isolated: bool


class SizeComplianceJudgeResult(JudgeResult[SizeComplianceChecks]):
    pass


# ---------------------------------------------------------------------
# Intrachunk Cohesion
# ---------------------------------------------------------------------


class IntrachunkCohesionChecks(StrictBaseModel):
    same_source_text: bool
    boundary_only_change: bool
    positive_single_topic: bool
    negative_mixes_distinct_topics: bool
    change_minimal: bool
    focus_valid: bool
    controlled_change_valid: bool
    metric_isolated: bool


class IntrachunkCohesionJudgeResult(
    JudgeResult[IntrachunkCohesionChecks],
):
    pass


# ---------------------------------------------------------------------
# HOPE Semantic Independence
# ---------------------------------------------------------------------


class HopeSemanticIndependenceChecks(StrictBaseModel):
    same_source_text: bool
    boundary_only_change: bool
    cue_question_valid: bool
    positive_self_contained: bool
    negative_has_context_dependency: bool
    missing_context_exists_elsewhere: bool
    focus_valid: bool
    controlled_change_valid: bool
    metric_isolated: bool


class HopeSemanticIndependenceJudgeResult(
    JudgeResult[HopeSemanticIndependenceChecks],
):
    pass


# ---------------------------------------------------------------------
# HOPE Information Preservation
# ---------------------------------------------------------------------


class HopeInformationPreservationChecks(StrictBaseModel):
    fact_exists_in_source: bool
    positive_preserves_fact: bool
    positive_preserves_source: bool
    negative_loses_or_distorts_fact: bool
    exactly_one_fact_affected: bool
    fact_not_recoverable_elsewhere: bool
    focus_valid: bool
    controlled_change_valid: bool
    change_minimal: bool


class HopeInformationPreservationJudgeResult(
    JudgeResult[HopeInformationPreservationChecks],
):
    pass


# ---------------------------------------------------------------------
# HOPE Concept Unity
# ---------------------------------------------------------------------


class HopeConceptUnityChecks(StrictBaseModel):
    same_source_text: bool
    boundary_only_change: bool
    positive_has_single_core_concept: bool
    negative_adds_independent_concept: bool
    added_content_is_not_merely_detail: bool
    change_minimal: bool
    focus_valid: bool
    controlled_change_valid: bool
    metric_isolated: bool


class HopeConceptUnityJudgeResult(
    JudgeResult[HopeConceptUnityChecks],
):
    pass


# ---------------------------------------------------------------------
# Contextual Coherence / DCC
# ---------------------------------------------------------------------


class ContextualCoherenceChecks(StrictBaseModel):
    same_source_text: bool
    boundary_only_change: bool
    local_structure_exists: bool
    positive_matches_local_context: bool
    negative_crosses_context_boundary: bool
    foreign_fragment_belongs_to_neighbor_context: bool
    change_minimal: bool
    focus_valid: bool
    controlled_change_valid: bool
    metric_isolated: bool


class ContextualCoherenceJudgeResult(
    JudgeResult[ContextualCoherenceChecks],
):
    pass


# ---------------------------------------------------------------------
# Boundary Clarity
# ---------------------------------------------------------------------


class BoundaryClarityChecks(StrictBaseModel):
    same_source_text: bool
    only_target_boundary_changed: bool
    positive_boundary_semantically_complete: bool
    negative_boundary_splits_dependency: bool
    negative_dependency_stronger: bool
    focus_valid: bool
    controlled_change_valid: bool
    metric_isolated: bool


class BoundaryClarityJudgeResult(
    JudgeResult[BoundaryClarityChecks],
):
    pass


# ---------------------------------------------------------------------
# ChunkScore
# ---------------------------------------------------------------------


class ChunkScoreChecks(StrictBaseModel):
    same_source_text: bool
    single_boundary_change: bool
    positive_chunk_complete: bool
    positive_boundary_clear: bool
    negative_chunk_less_complete: bool
    negative_boundary_less_clear: bool
    same_change_causes_both_effects: bool
    focus_valid: bool
    controlled_change_valid: bool
    no_extra_violation: bool


class ChunkScoreJudgeResult(
    JudgeResult[ChunkScoreChecks],
):
    pass
