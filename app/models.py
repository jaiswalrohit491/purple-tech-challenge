from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


EventType = Literal[
    "ENTRY",
    "EXIT",
    "ZONE_ENTER",
    "ZONE_EXIT",
    "ZONE_DWELL",
    "BILLING_QUEUE_JOIN",
    "BILLING_QUEUE_ABANDON",
    "REENTRY",
]


class EventMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")
    queue_depth: int | None = None
    sku_zone: str | None = None
    session_seq: int | None = None


class Event(BaseModel):
    event_id: UUID
    store_id: str = Field(min_length=1, max_length=64)
    camera_id: str = Field(min_length=1, max_length=64)
    visitor_id: str = Field(min_length=1, max_length=64)
    event_type: EventType
    timestamp: datetime
    zone_id: str | None = None
    dwell_ms: int = Field(default=0, ge=0)
    is_staff: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: EventMetadata = EventMetadata()


class IngestBatch(BaseModel):
    events: list[Event]


class IngestError(BaseModel):
    index: int
    event_id: str | None
    error: str


class IngestResponse(BaseModel):
    accepted: int
    duplicates: int
    rejected: int
    errors: list[IngestError] = []


class ZoneDwell(BaseModel):
    zone_id: str
    avg_dwell_ms: float
    visits: int


class StoreMetrics(BaseModel):
    store_id: str
    window_start: datetime
    window_end: datetime
    unique_visitors: int
    staff_count: int
    # CONFIRMED conversions (brief's definition): visitor was in billing
    # zone within 5 min of a POS. Weakest of the three signals — counts
    # people near the till who may or may not have actually paid.
    converted_visitors: int
    conversion_rate: float
    # VERIFIED purchases (strongest signal): full ENTRY→BILLING→EXIT
    # trajectory with quick exit and POS between billing and exit. Rules
    # out browsers who happen to be near the till. Always <= confirmed.
    verified_purchases: int
    verified_purchase_rate: float
    # POTENTIAL conversions (inferred): visitor browsed a brand zone + that
    # brand sold within `pos_correlation_window`. Correlation, not
    # causation — off-camera customers may be the real buyers. Always
    # >= confirmed.
    potential_converted: int
    potential_conversion_rate: float
    avg_dwell_by_zone: list[ZoneDwell]
    current_queue_depth: int
    abandonment_rate: float


class FunnelStage(BaseModel):
    stage: str
    visitors: int
    dropoff_pct: float


class StoreFunnel(BaseModel):
    store_id: str
    window_start: datetime
    window_end: datetime
    stages: list[FunnelStage]


class HeatmapZone(BaseModel):
    zone_id: str
    visit_count: int
    avg_dwell_ms: float
    visit_score: float
    dwell_score: float


class StoreHeatmap(BaseModel):
    store_id: str
    window_start: datetime
    window_end: datetime
    zones: list[HeatmapZone]
    data_confidence: Literal["LOW", "OK"]


class Anomaly(BaseModel):
    anomaly_type: Literal["BILLING_QUEUE_SPIKE", "CONVERSION_DROP", "DEAD_ZONE"]
    severity: Literal["INFO", "WARN", "CRITICAL"]
    detected_at: datetime
    detail: dict
    suggested_action: str


class StoreAnomalies(BaseModel):
    store_id: str
    anomalies: list[Anomaly]


class StoreFeedHealth(BaseModel):
    store_id: str
    last_event_at: datetime | None
    stale: bool


class HealthResponse(BaseModel):
    status: Literal["OK", "DEGRADED"]
    version: str
    db_reachable: bool
    stores: list[StoreFeedHealth]
