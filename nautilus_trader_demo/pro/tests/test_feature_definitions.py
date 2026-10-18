"""新增特征只改JSON：不同字段/频率/业务键共享Provider、缓存与文件契约。"""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from bomber.framework.datahub import DataHub
from bomber.framework.datahub.features import (FeatureError, FeatureHub, FeatureQuery, FeatureRequirement,
    FeatureUnavailable, dataset_from_definition)
from bomber.framework.dataprep.features import (DEFAULT_FEATURE_CONFIG, bundle_from_definition,
    DolphinDbRawFeatureProvider, FeatureQueryCache, FeatureRequest, FileFeatureProvider, export_features)
from bomber.framework.dataprep.sources.policy import DataSourcePurpose, validate_source
from bomber.framework.dataprep.sources.dolphindb import DolphinDbReferenceConfig
from tests.test_dolphindb_index_features import END, M, Session, row

EXAMPLE = DEFAULT_FEATURE_CONFIG.with_name("features.example.json")
B_END = END//(5*M)*(5*M)


class GeneralSession(Session):
    def __init__(self, kind="basis", rows=None):
        fields = {"instrument": "STRING", "eventTime": "LONG"}
        if kind == "basis":
            fields.update(publishedTime="LONG", revision="LONG", quality="INT",
                basis_value="DOUBLE", lookback="INT", currency="STRING")
            default = dict(key="RB.SHFE", event_ns=B_END, published_ns=B_END+100,
                revision_ns=1, quality_flags=0, p_basis=-25.5, p_lookback=5, p_currency=None)
        else:
            fields.update(annual_sigma="DOUBLE", valid="BOOL")
            default = dict(key="asset:IM", event_ns=END, published_ns=0, p_sigma=.25, p_valid=True)
        super().__init__([default] if rows is None else rows,
            pd.DataFrame([dict(name=name, typeString=dtype) for name, dtype in fields.items()]))
        self.columns = tuple(default)

    def run(self, script):
        if script.startswith("schema("):
            return super().run(script)
        self.scripts.append(script)
        return pd.DataFrame(self.rows, columns=self.columns)


class DefinitionTests(unittest.TestCase):
    def setUp(self):
        self.connection = DolphinDbReferenceConfig("fixture", 8848, "user", "secret")
        self.definition = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        self.now = END+1000

    def bundle(self, definition=None):
        return bundle_from_definition(definition or self.definition, self.connection)

    def provider(self, bundle, dataset, session):
        provider = bundle.provider(dataset, session_factory=lambda **kwargs: session, clock_ns=lambda: self.now)
        self.addCleanup(provider.close)
        provider.open()
        return provider

    def test_two_new_features_only_json_same_provider_type_and_distinct_payloads(self):
        bundle = self.bundle()
        basis = self.provider(bundle, "basis.five_minute", GeneralSession())
        vol = self.provider(bundle, "volatility.minute", GeneralSession("vol"))
        self.assertIs(type(basis), DolphinDbRawFeatureProvider)
        self.assertIs(type(vol), type(basis))
        first, = basis.read(FeatureRequest("basis.five_minute", 1, ("RB.SHFE",), B_END, B_END, self.now)).records
        second, = vol.read(FeatureRequest("volatility.minute", 1, ("asset:IM",), END, END, self.now)).records
        self.assertEqual(first.payload.basis, Decimal("-25.5"))
        self.assertEqual(first.payload.lookback, 5)
        self.assertIsNone(first.payload.currency)
        self.assertEqual(second.payload.sigma, .25)
        self.assertIs(second.payload.valid, True)
        self.assertIsNot(type(first.payload), type(second.payload))

    def test_generated_payload_strict_types_frozen_slots_and_unknown_fields(self):
        spec = self.bundle().registry.get("basis.five_minute", 1)
        payload = spec.payload_type(basis=Decimal(1), lookback=5, currency="CNY")
        with self.assertRaises(FrozenInstanceError):
            payload.basis = Decimal(2)
        self.assertFalse(hasattr(payload, "__dict__"))
        with self.assertRaises(FeatureError):
            spec.payload_type(basis=1., lookback=5, currency="CNY")
        with self.assertRaises(FeatureError):
            spec.payload_type(basis=Decimal(1), lookback=0, currency="CNY")
        with self.assertRaises(TypeError):
            spec.payload_type(basis=Decimal(1), lookback=5, currency=None, extra=1)

    def test_reload_schema_same_type_and_fingerprint_constraints_change_version_identity(self):
        spec = self.bundle().registry.get("volatility.minute", 1)
        reloaded = self.bundle().registry.get("volatility.minute", 1)
        self.assertIs(spec.payload_type, reloaded.payload_type)
        self.assertEqual(spec.fingerprint, reloaded.fingerprint)
        changed = deepcopy(self.definition)
        changed["datasets"][1]["schema"]["fields"][0]["maximum"] = ".5"
        self.assertNotEqual(spec.fingerprint, self.bundle(changed).registry.get("volatility.minute", 1).fingerprint)
        changed = deepcopy(self.definition)
        changed["datasets"][1]["schema"]["frequency"]["period_ns"] = 5*M
        different = self.bundle(changed).registry.get("volatility.minute", 1)
        with self.assertRaises(FeatureError):
            different.validate_payload(spec.payload_type(sigma=.25, valid=True))

    def test_json_unknown_keys_executable_types_invalid_relations_rejected(self):
        schema = deepcopy(self.definition["datasets"][1]["schema"])
        for changes in ({"python": "eval(...)"}, {"relations": [{"left": "sigma", "op": "eval", "right": "sigma"}]}):
            with self.assertRaises(FeatureError):
                dataset_from_definition({**schema, **changes})
        schema["fields"][0]["kind"] = "some.module.Class"
        with self.assertRaises(FeatureError):
            dataset_from_definition(schema)
        duplicate = deepcopy(self.definition)
        duplicate["datasets"].append(deepcopy(duplicate["datasets"][0]))
        with self.assertRaises(FeatureError):
            self.bundle(duplicate)

    def test_five_minute_start_label_uses_declared_period_without_index_assumptions(self):
        bundle, session = self.bundle(), GeneralSession()
        config = bundle.binding("basis.five_minute").source_config
        config = replace(config, event_time=replace(config.event_time, label="start"))
        provider = DolphinDbRawFeatureProvider(config, bundle.registry,
            session_factory=lambda **kwargs: session, clock_ns=lambda: self.now)
        self.addCleanup(provider.close)
        provider.open()
        provider.read(FeatureRequest("basis.five_minute", 1, ("RB.SHFE",), B_END, B_END, self.now))
        self.assertIn("+300000000000l", session.scripts[-1])
        self.assertNotIn("concatDateTime", session.scripts[-1])
        original = self.provider(bundle, "basis.five_minute", session)
        request = FeatureRequest("basis.five_minute", 1, ("RB.SHFE",), B_END, B_END, self.now)
        cursor = original.read_updates(request).cursor
        with self.assertRaisesRegex(FeatureError, "来源/范围改变"):
            provider.read_updates(request, cursor)

    def test_multi_feature_cache_same_datahub_exact_index_and_asof_basis(self):
        combined = json.loads(DEFAULT_FEATURE_CONFIG.read_text(encoding="utf-8"))
        combined["datasets"].extend(self.definition["datasets"])
        bundle = self.bundle(combined)
        index = bundle.provider("index.minute", session_factory=lambda **kwargs: Session([row()]), clock_ns=lambda: self.now)
        basis = bundle.provider("basis.five_minute", session_factory=lambda **kwargs: GeneralSession(), clock_ns=lambda: self.now)
        hub = DataHub(features=FeatureHub(bundle.registry))
        first = FeatureQueryCache(index, dataset_id="index.minute", schema_version=1, keys=("000852",),
            start_ns=END, end_ns=END, clock_ns=lambda: self.now, hub=hub, source_id="one")
        second = FeatureQueryCache(basis, dataset_id="basis.five_minute", schema_version=1, keys=("RB.SHFE",),
            start_ns=B_END, end_ns=B_END, clock_ns=lambda: self.now, hub=hub, source_id="two")
        self.addCleanup(first.close)
        self.addCleanup(second.close)
        first.open()
        second.open()
        self.assertIs(first.hub, second.hub)
        snapshot = hub.feature_snapshot((FeatureRequirement("index", "index.minute", 1, "000852"),
            FeatureRequirement("basis", "basis.five_minute", 1, "RB.SHFE")), FeatureQuery(END, self.now))
        self.assertEqual(snapshot.latest("index").payload.close, Decimal("6401.2"))
        self.assertEqual(snapshot.latest("basis").payload.basis, Decimal("-25.5"))
        self.assertEqual(first.manifest["cached_records"], 1)
        self.assertEqual(second.manifest["cached_records"], 1)

    def test_explicit_revision_quality_and_no_overwrite_of_same_revision(self):
        bundle, session = self.bundle(), GeneralSession()
        provider = self.provider(bundle, "basis.five_minute", session)
        request = FeatureRequest("basis.five_minute", 1, ("RB.SHFE",), B_END, B_END, self.now)
        first = provider.read_updates(request)
        session.rows[0]["p_basis"] = -24.5
        with self.assertRaisesRegex(FeatureError, "原地覆盖"):
            provider.read_updates(request, first.cursor)
        session.rows[0].update(revision_ns=2, published_ns=B_END+200, quality_flags=1)
        second = provider.read_updates(request, first.cursor)
        hub = FeatureHub(bundle.registry)
        hub.store.publish((*first.records, *second.records))
        with self.assertRaises(FeatureUnavailable) as error:
            hub.snapshot((FeatureRequirement("b", "basis.five_minute", 1, "RB.SHFE"),), FeatureQuery(END, self.now))
        self.assertEqual(error.exception.code, "QUALITY")

    def test_generated_schema_file_roundtrip_reload_and_offline_config(self):
        bundle = self.bundle()
        provider = self.provider(bundle, "basis.five_minute", GeneralSession())
        request = FeatureRequest("basis.five_minute", 1, ("RB.SHFE",), B_END, B_END, self.now)
        rows = provider.read(request).records
        with TemporaryDirectory() as temporary:
            path = Path(temporary)/"basis.csv"
            export_features(path, bundle.registry, rows)
            offline = deepcopy(self.definition)
            offline["datasets"] = offline["datasets"][:1]
            offline["datasets"][0]["source"] = {"backend": "file", "path": str(path)}
            rebuilt = bundle_from_definition(offline, {})
            with rebuilt.provider("basis.five_minute") as file:
                self.assertIs(type(file), FileFeatureProvider)
                self.assertEqual(file.read(request).records, rows)
            with self.assertRaises(ValueError):
                rebuilt.provider("basis.five_minute", purpose=DataSourcePurpose.LIVE_FEATURE)

    def test_mongodb_binding_is_configuration_with_credentials_injected(self):
        source = deepcopy(self.definition)
        source["datasets"] = source["datasets"][:1]
        columns = {"payload.basis": "basis", "payload.lookback": "lookback", "payload.currency": "currency"}
        source["datasets"][0]["source"] = {"backend": "mongodb", "database": "features", "table": "values",
            "source_id": "fixture-mongo", "mapping": {"columns": columns}}
        bundle = bundle_from_definition(source, {"mongodb": {"uri": "mongodb://fixture"}})
        provider = bundle.provider("basis.five_minute")
        self.assertEqual(provider.config.source_id, "fixture-mongo")
        self.assertNotIn("mongodb://fixture", repr(provider.config))

    def test_declared_calendar_requires_injected_slots_and_source_policy_accepts_raw_format(self):
        schema = deepcopy(self.definition["datasets"][0]["schema"])
        schema["frequency"]["calendar_id"] = "session-five-minute"
        with self.assertRaises(FeatureError):
            dataset_from_definition(schema)
        spec = dataset_from_definition(schema, calendars={"session-five-minute": lambda start, end: (end,)})
        self.assertEqual(spec.frequency.expected_ends(B_END-5*M, B_END), (B_END,))
        self.assertEqual(validate_source("dolphindb-raw", DataSourcePurpose.LIVE_FEATURE), DataSourcePurpose.LIVE_FEATURE)


if __name__ == "__main__":
    unittest.main()
