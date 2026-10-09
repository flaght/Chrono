"""在线期货参考资料时效；来源日不使用自然日减一推断交易日。"""
from dataclasses import dataclass
from datetime import date

from bomber.framework.datahub.sector_roles import SectorDataUnavailable


@dataclass(frozen=True)
class ReferenceFreshnessPolicy:
    expected_source_day: date | None = None
    max_source_age_days: int = 14
    max_observation_age_seconds: int = 30
    max_refresh_duration_seconds: int = 30

    def __post_init__(self):
        if self.expected_source_day is not None and type(self.expected_source_day) is not date:
            raise TypeError("预期来源日须为date")
        for name in ("max_source_age_days", "max_observation_age_seconds", "max_refresh_duration_seconds"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name}须为正整数")

    def validate_day(self, source_day, trading_day):
        age = (trading_day - source_day).days
        if age <= 0 or age > self.max_source_age_days:
            raise SectorDataUnavailable(f"参考资料来源日过期或非此前日期: source={source_day} "
                f"TD={trading_day} age_days={age} max={self.max_source_age_days}")
        if self.expected_source_day is not None and source_day != self.expected_source_day:
            raise SectorDataUnavailable(f"角色资料来源日不符: expected={self.expected_source_day} actual={source_day}")

    def validate_observation(self, observed_ns, now_ns):
        age = now_ns - observed_ns
        if not 0 <= age <= self.max_observation_age_seconds * 1_000_000_000:
            raise SectorDataUnavailable("参考资料最近成功观测已过期或时钟回退")

    def manifest(self):
        return {"expected_source_day": str(self.expected_source_day) if self.expected_source_day else None,
            "role_and_factor_max_source_age_days": self.max_source_age_days,
            "basic_age_basis": "contract_lifecycle_and_successful_observation_not_listing_date",
            "max_observation_age_seconds": self.max_observation_age_seconds,
            "max_refresh_duration_seconds": self.max_refresh_duration_seconds,
            "source_day_evidence": "operator_declared_day" if self.expected_source_day else
                "bounded_age_only_not_exchange_calendar_completeness"}
