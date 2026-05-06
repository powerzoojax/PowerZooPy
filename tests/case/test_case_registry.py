"""Tests for the case registry and metadata system."""

import warnings

import pytest

from powerzoo.case import load_case, list_cases
from powerzoo.case._registry import CaseMeta
from powerzoo.case.CaseBase import ClearCase


# ------------------------------------------------------------------
# Metadata completeness
# ------------------------------------------------------------------

class TestMetadataCompleteness:
    """Every Case class must have non-empty GRID_TYPE and BUS_COUNT."""

    def test_all_cases_have_grid_type(self):
        for meta in list_cases():
            assert meta.grid_type in ("transmission", "distribution"), (
                f"{meta.name} has invalid GRID_TYPE='{meta.grid_type}'"
            )

    def test_all_cases_have_bus_count(self):
        for meta in list_cases():
            assert meta.bus_count > 0, f"{meta.name} has BUS_COUNT={meta.bus_count}"

    def test_all_cases_have_description(self):
        for meta in list_cases():
            assert meta.description, f"{meta.name} has empty DESCRIPTION"


# ------------------------------------------------------------------
# Directory / grid_type consistency
# ------------------------------------------------------------------

class TestDirectoryConsistency:
    """Cases under transmission/ must be transmission, etc."""

    def test_transmission_cases_have_correct_type(self):
        for meta in list_cases(grid_type="transmission"):
            assert "transmission" in meta.module_path, (
                f"{meta.name} is grid_type='transmission' but module_path={meta.module_path}"
            )

    def test_distribution_cases_have_correct_type(self):
        for meta in list_cases(grid_type="distribution"):
            assert "distribution" in meta.module_path, (
                f"{meta.name} is grid_type='distribution' but module_path={meta.module_path}"
            )


# ------------------------------------------------------------------
# list_cases filtering
# ------------------------------------------------------------------

class TestListCases:
    def test_returns_all(self):
        assert len(list_cases()) >= 14

    def test_filter_by_grid_type(self):
        trans = list_cases(grid_type="transmission")
        assert all(m.grid_type == "transmission" for m in trans)
        assert len(trans) >= 8

    def test_filter_by_min_buses(self):
        big = list_cases(min_buses=300)
        assert all(m.bus_count >= 300 for m in big)

    def test_filter_by_phase(self):
        three_phase = list_cases(phase="3")
        assert len(three_phase) >= 1
        assert all(m.phase == "3" for m in three_phase)

    def test_sorted_by_bus_count(self):
        cases = list_cases()
        counts = [m.bus_count for m in cases]
        assert counts == sorted(counts)


# ------------------------------------------------------------------
# load_case with new sub-directory layout
# ------------------------------------------------------------------

class TestLoadCase:
    def test_load_by_int(self):
        c = load_case(5)
        assert c is not None
        assert len(c.nodes) == 5

    def test_load_by_name(self):
        c = load_case("Case33bw")
        assert c is not None
        assert len(c.nodes) == 33

    def test_load_with_grid_type_hint(self):
        c = load_case("Case5", grid_type="transmission")
        assert c is not None

    def test_load_wrong_grid_type_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            c = load_case("Case33bw", grid_type="transmission")
            assert c is not None
            assert any("GRID_TYPE" in str(x.message) for x in w)


# ------------------------------------------------------------------
# Convenience re-exports from powerzoo.case
# ------------------------------------------------------------------

class TestReExports:
    def test_import_from_case_package(self):
        from powerzoo.case import Case5, Case33bw
        assert issubclass(Case5, ClearCase)
        assert issubclass(Case33bw, ClearCase)

    def test_import_from_subpackage(self):
        from powerzoo.case.transmission import Case5
        from powerzoo.case.distribution import Case33bw
        assert Case5.GRID_TYPE == "transmission"
        assert Case33bw.GRID_TYPE == "distribution"
