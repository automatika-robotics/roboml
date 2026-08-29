"""CI-safe tests: RoboBrain family detection, task prompts, and output extraction."""

import logging

from roboml.models.planning import (
    FAMILY_ROBOBRAIN20,
    FAMILY_ROBOBRAIN25,
    _build_task_text,
    _detect_family,
    _extract_structured_output,
    _normalize_output,
    _supports_thinking,
)

logger = logging.getLogger("test")

FAMILIES = (FAMILY_ROBOBRAIN20, FAMILY_ROBOBRAIN25)


class TestFamilyDetection:
    def test_qwen2_5_vl_is_robobrain20(self):
        assert _detect_family("qwen2_5_vl", logger) == FAMILY_ROBOBRAIN20

    def test_qwen3_vl_is_robobrain25(self):
        assert _detect_family("qwen3_vl", logger) == FAMILY_ROBOBRAIN25

    def test_unknown_defaults_to_robobrain25(self):
        assert _detect_family("some_future_vlm", logger) == FAMILY_ROBOBRAIN25


class TestThinkingSupport:
    def test_robobrain20_3b_has_no_thinking(self):
        assert _supports_thinking(FAMILY_ROBOBRAIN20, "BAAI/RoboBrain2.0-3B") is False

    def test_robobrain20_7b_has_thinking(self):
        assert _supports_thinking(FAMILY_ROBOBRAIN20, "BAAI/RoboBrain2.0-7B") is True

    def test_robobrain25_has_no_thinking(self):
        assert _supports_thinking(FAMILY_ROBOBRAIN25, "BAAI/RoboBrain2.5-4B") is False
        assert (
            _supports_thinking(FAMILY_ROBOBRAIN25, "BAAI/RoboBrain2.5-8B-NV") is False
        )


class TestTaskPrompts:
    def test_general_passthrough_in_both_families(self):
        for family in FAMILIES:
            assert _build_task_text("hello", "general", family) == "hello"

    def test_grounding_identical_across_families(self):
        text20 = _build_task_text("the red cup", "grounding", FAMILY_ROBOBRAIN20)
        text25 = _build_task_text("the red cup", "grounding", FAMILY_ROBOBRAIN25)
        assert text20 == text25
        assert "bounding box coordinate" in text20

    def test_pointing_prompts_differ_per_family(self):
        text20 = _build_task_text("the pickles", "pointing", FAMILY_ROBOBRAIN20)
        text25 = _build_task_text("the pickles", "pointing", FAMILY_ROBOBRAIN25)
        assert "normalized" in text20
        assert "2D coordinates" in text25

    def test_affordance_is_pointing_in_robobrain25(self):
        affordance25 = _build_task_text(
            "hold the cup", "affordance", FAMILY_ROBOBRAIN25
        )
        pointing25 = _build_task_text("hold the cup", "pointing", FAMILY_ROBOBRAIN25)
        assert affordance25 == pointing25
        # 2.0 keeps its own affordance prompt
        affordance20 = _build_task_text(
            "hold the cup", "affordance", FAMILY_ROBOBRAIN20
        )
        assert "affordance area" in affordance20

    def test_trajectory_asks_for_depth_in_robobrain25(self):
        text25 = _build_task_text("reach the banana", "trajectory", FAMILY_ROBOBRAIN25)
        assert "3D" in text25
        assert "depth" in text25
        text20 = _build_task_text("reach the banana", "trajectory", FAMILY_ROBOBRAIN20)
        assert "depth" not in text20


class TestOutputExtraction:
    def test_pointing_points_in_both_families(self):
        for family in FAMILIES:
            out = _extract_structured_output("[(10, 20), (30, 40)]", "pointing", family)
            assert out == [(10, 20), (30, 40)]

    def test_grounding_boxes_in_both_families(self):
        for family in FAMILIES:
            out = _extract_structured_output(
                "The box is [10, 20, 30, 40]", "grounding", family
            )
            assert out == [[10, 20, 30, 40]]

    def test_trajectory_3d_in_robobrain25(self):
        out = _extract_structured_output(
            "[(10, 20, 0.5), (30, 40, 0.75)]", "trajectory", FAMILY_ROBOBRAIN25
        )
        assert out == [[(10, 20, 0.5), (30, 40, 0.75)]]

    def test_trajectory_integer_depth_in_robobrain25(self):
        # regression: integer depths must not be silently dropped
        out = _extract_structured_output(
            "[(100, 200, 1), (300, 400, 2)]", "trajectory", FAMILY_ROBOBRAIN25
        )
        assert out == [[(100, 200, 1.0), (300, 400, 2.0)]]

    def test_trajectory_mixed_depth_in_robobrain25(self):
        # regression: mixed integer/decimal depths must not truncate the
        # trajectory
        out = _extract_structured_output(
            "[(100, 200, 1), (300, 400, 2.0)]", "trajectory", FAMILY_ROBOBRAIN25
        )
        assert out == [[(100, 200, 1.0), (300, 400, 2.0)]]

    def test_trajectory_bracket_tuples_in_robobrain25(self):
        out = _extract_structured_output(
            "[[10, 20, 0.5], [30, 40, 2]]", "trajectory", FAMILY_ROBOBRAIN25
        )
        assert out == [[(10, 20, 0.5), (30, 40, 2.0)]]

    def test_trajectory_ignores_box_style_output_in_robobrain25(self):
        # a grounding-style box must not be misread as a single 3D waypoint
        out = _extract_structured_output(
            "[100, 200, 300, 400]", "trajectory", FAMILY_ROBOBRAIN25
        )
        assert out == [[]]

    def test_trajectory_2d_in_robobrain20(self):
        out = _extract_structured_output(
            "[[10, 20], [30, 40]]", "trajectory", FAMILY_ROBOBRAIN20
        )
        assert out == [[(10, 20), (30, 40)]]

    def test_affordance_boxes_in_20_points_in_25(self):
        out20 = _extract_structured_output(
            "[10, 20, 30, 40]", "affordance", FAMILY_ROBOBRAIN20
        )
        assert out20 == [[10, 20, 30, 40]]
        out25 = _extract_structured_output(
            "[(15, 25)]", "affordance", FAMILY_ROBOBRAIN25
        )
        assert out25 == [(15, 25)]

    def test_general_returns_raw_text(self):
        for family in FAMILIES:
            assert _extract_structured_output("free text", "general", family) == (
                "free text"
            )


class TestOutputNormalization:
    """RoboBrain 2.5 outputs are brought to the 2.0 contract: pixel
    coordinates of the input image, boxes for affordance, 2D waypoints."""

    SIZE = (640, 480)

    def test_robobrain20_passes_through_untouched(self):
        for task, raw in (
            ("pointing", [(10, 20)]),
            ("grounding", [[10, 20, 30, 40]]),
            ("affordance", [[10, 20, 30, 40]]),
            ("trajectory", [[(10, 20), (30, 40)]]),
        ):
            out, extras = _normalize_output(raw, task, FAMILY_ROBOBRAIN20, self.SIZE)
            assert out == raw and extras == {}

    def test_text_and_empty_outputs_pass_through(self):
        assert _normalize_output(
            "free text", "general", FAMILY_ROBOBRAIN25, self.SIZE
        ) == ("free text", {})
        assert _normalize_output([], "pointing", FAMILY_ROBOBRAIN25, self.SIZE) == (
            [],
            {},
        )

    def test_unknown_image_size_leaves_coordinates_alone(self):
        out, _ = _normalize_output([(500, 500)], "pointing", FAMILY_ROBOBRAIN25, None)
        assert out == [(500, 500)]

    def test_robobrain25_points_become_pixels(self):
        out, extras = _normalize_output(
            [(500, 500), (0, 0)], "pointing", FAMILY_ROBOBRAIN25, self.SIZE
        )
        assert out == [(320, 240), (0, 0)] and extras == {}

    def test_robobrain25_pixels_are_clamped_to_the_image(self):
        out, _ = _normalize_output(
            [(1000, 1000)], "pointing", FAMILY_ROBOBRAIN25, self.SIZE
        )
        assert out == [(639, 479)]

    def test_robobrain25_boxes_become_pixels(self):
        out, _ = _normalize_output(
            [[250, 250, 750, 750]], "grounding", FAMILY_ROBOBRAIN25, self.SIZE
        )
        assert out == [[160, 120, 480, 360]]

    def test_robobrain25_affordance_point_becomes_a_box_around_it(self):
        out, _ = _normalize_output(
            [(500, 500)], "affordance", FAMILY_ROBOBRAIN25, self.SIZE
        )
        # 5% of the shorter side (480) is 24 px: a 12 px half-side box
        # centered on the pixel (320, 240)
        assert out == [[308, 228, 332, 252]]

    def test_robobrain25_affordance_box_stays_inside_the_image(self):
        out, _ = _normalize_output(
            [(0, 0)], "affordance", FAMILY_ROBOBRAIN25, self.SIZE
        )
        assert out == [[0, 0, 12, 12]]

    def test_failed_trajectory_parse_gains_no_depths(self):
        # [[]] is what a failed trajectory extraction returns; it must not
        # come back advertising an (empty) depths field
        out, extras = _normalize_output(
            [[]], "trajectory", FAMILY_ROBOBRAIN25, self.SIZE
        )
        assert out == [[]] and extras == {}

    def test_robobrain25_trajectory_is_2d_with_depths_apart(self):
        out, extras = _normalize_output(
            [[(500, 500, 0.5), (1000, 0, 2.0)]],
            "trajectory",
            FAMILY_ROBOBRAIN25,
            self.SIZE,
        )
        assert out == [[(320, 240), (639, 0)]]
        assert extras == {"depths": [[0.5, 2.0]]}
