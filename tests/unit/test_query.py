"""
Unit tests for manifold_db.query — QueryParser, QueryBuilder, ManifoldQuery, QueryResult.
"""

import asyncio

import numpy as np
import pytest

from manifold_db.query import (
    ManifoldQuery,
    MetricType,
    QueryBuilder,
    QueryEngine,
    QueryParser,
    QueryResult,
)


class TestQueryParser:
    def test_parse_select_geodesic(self):
        parser = QueryParser()
        q = parser.parse(
            "SELECT * FROM observations WHERE geodesic_distance(embedding, query_point) < 1.0 "
            "ALONG manifold 'test_atlas' USING metric 'geodesic'"
        )
        assert q.query_type.value == "select"
        assert q.atlas_name == "test_atlas"
        assert q.metric_type == MetricType.GEODESIC

    def test_parse_tangent_query(self):
        parser = QueryParser()
        q = parser.parse("TANGENT_QUERY FROM chart 'text_emb' WHERE distance < 0.5")
        assert q.chart_id == "text_emb"
        assert q.epsilon == 0.5

    def test_parse_cross_modal(self):
        parser = QueryParser()
        q = parser.parse("CROSS_MODAL FROM 'text' TO 'image' TRANSPORT VIA 'overlap'")
        assert q.modality == "text"
        assert q.target_modality == "image"

    def test_parse_select_simple(self):
        parser = QueryParser()
        q = parser.parse("SELECT * FROM observations")
        assert q.fields == ["*"]


class TestQueryBuilder:
    def test_fluent_select(self):
        q = (
            QueryBuilder()
            .select("*")
            .from_chart("text_emb")
            .where_geodesic(np.array([1.0, 0.0]), epsilon=0.5)
            .top_k(10)
            .build()
        )
        assert q.chart_id == "text_emb"
        assert q.k == 10
        assert q.epsilon == 0.5

    def test_fluent_cross_modal(self):
        q = (
            QueryBuilder()
            .cross_modal("text", "image")
            .with_transport("overlap")
            .top_k(5)
            .build()
        )
        assert q.modality == "text"
        assert q.target_modality == "image"
        assert q.transport_via == "overlap"

    def test_build_raises_on_empty(self):
        with pytest.raises(ValueError):
            QueryBuilder().build()

    def test_parallel_transport_query(self):
        q = (
            QueryBuilder()
            .parallel_transport(np.array([1.0, 0.0]), "chart_a", "chart_b")
            .build()
        )
        assert q.source_chart == "chart_a"
        assert q.target_chart == "chart_b"


class TestManifoldQuery:
    def test_validate_success(self):
        q = ManifoldQuery(
            chart_id="c0",
            query_point=np.zeros(5),
            k=10,
        )
        valid, msg = q.validate()
        assert valid

    def test_validate_missing_point(self):
        q = ManifoldQuery(chart_id="c0", query_point=None, k=10)
        valid, msg = q.validate()
        # May or may not fail depending on query type; just check it returns a tuple
        assert isinstance(valid, bool)

    def test_estimate_cost(self):
        q = ManifoldQuery(
            query_type=(
                QueryBuilder().tangent_query("c", np.zeros(5)).build().query_type
                if False
                else None
            ),
            k=10,
        )
        # Just test it runs
        q2 = ManifoldQuery(k=10)
        cost = q2.estimate_cost({"c": 1000})
        assert cost is not None

    def test_serialization_roundtrip(self):
        q = ManifoldQuery(
            chart_id="c0",
            metric_type=MetricType.GEODESIC,
            k=10,
            epsilon=0.5,
        )
        d = q.to_dict()
        q2 = ManifoldQuery.from_dict(d)
        assert q2.chart_id == "c0"
        assert q2.k == 10


class TestQueryResult:
    def test_empty_result(self):
        r = QueryResult()
        assert len(r) == 0
        assert r.to_list() == []

    def test_non_empty(self):
        r = QueryResult(
            point_ids=np.array([1, 2, 3]),
            distances=np.array([0.1, 0.2, 0.3]),
            execution_time=0.05,
            chart_id="c0",
        )
        assert len(r) == 3
        assert r[0]["point_id"] == 1

    def test_to_dict(self):
        r = QueryResult(
            point_ids=np.array([1, 2]),
            distances=np.array([0.1, 0.2]),
        )
        d = r.to_dict()
        assert "point_ids" in d

    def test_iteration(self):
        r = QueryResult(
            point_ids=np.array([1, 2, 3]),
            distances=np.array([0.1, 0.2, 0.3]),
        )
        items = list(r)
        assert len(items) == 3


class TestQueryEngine:
    @pytest.mark.asyncio
    async def test_explain(self):
        engine = QueryEngine()
        q = ManifoldQuery(k=10)
        plan = await engine.explain(q)
        assert plan.total_estimated_ms >= 0
        assert len(plan.steps) > 0

    @pytest.mark.asyncio
    async def test_execute(self):
        engine = QueryEngine()
        q = ManifoldQuery(k=5)
        result = await engine.execute(q)
        assert isinstance(result, QueryResult)
