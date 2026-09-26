"""Event-triggered front-end for the frozen DINO anchor teacher.

The expensive teacher remains the source of truth.  A DINO refresh stores
small role-specific RGB templates.  Intermediate frames use local weighted
zero-normalized cross correlation (ZNCC), velocity prediction, and the known
two-link geometry.  Unreliable cheap proposals are discarded before they can
enter replay.
"""

import time
from collections import Counter

import numpy as np
import torch


ROLES = ("base", "elbow", "control_tip", "goal")


def _value(config, key, default):
    try:
        return config.get(key, default)
    except AttributeError:
        return getattr(config, key, default)


def scheduled_refresh_reasons(is_first, since_refresh, max_interval):
    """Pure scheduling rule used by the wrapper and CPU-only contracts."""
    resets = np.asarray(is_first, dtype=bool).reshape(-1)
    intervals = np.asarray(since_refresh, dtype=np.int64).reshape(-1)
    reasons = set()
    if resets.any():
        reasons.add("reset")
    if (intervals >= int(max_interval)).any() and not resets.any():
        reasons.add("max_interval")
    return reasons


def _extract_patch(image, center, radius):
    """Extract a channel-first patch, or ``None`` at an image boundary."""
    height, width = image.shape[:2]
    x, y = np.rint(center).astype(np.int64)
    if x - radius < 0 or x + radius >= width:
        return None
    if y - radius < 0 or y + radius >= height:
        return None
    patch = image[y - radius:y + radius + 1, x - radius:x + radius + 1]
    return np.ascontiguousarray(patch.transpose(2, 0, 1), dtype=np.float32) / 255.0


def _template_weight(template):
    """Centre-weight foreground pixels so changing video does not dominate."""
    _, size, _ = template.shape
    coordinates = np.arange(size, dtype=np.float32) - 0.5 * (size - 1)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    sigma = max(0.24 * size, 1.0)
    spatial = np.exp(-(np.square(xx) + np.square(yy)) / (2.0 * sigma**2))

    border = np.zeros((size, size), dtype=bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    background = np.median(template[:, border], axis=1)[:, None, None]
    contrast = np.linalg.norm(template - background, axis=0)
    scale = max(float(np.percentile(contrast, 80.0)), 0.03)
    foreground = np.clip(contrast / scale, 0.0, 1.0)
    weight = spatial * (0.15 + 0.85 * foreground)
    return np.ascontiguousarray(weight / max(float(weight.sum()), 1e-6), dtype=np.float32)


def _weighted_zncc(template, candidate, weight):
    """Per-channel weighted ZNCC, aggregated over the small RGB patch."""
    if not (
        np.isfinite(template).all()
        and np.isfinite(candidate).all()
        and np.isfinite(weight).all()
    ):
        return float("nan")
    denominator = max(float(weight.sum()), 1e-8)
    template_mean = (template * weight[None]).sum(axis=(1, 2)) / denominator
    candidate_mean = (candidate * weight[None]).sum(axis=(1, 2)) / denominator
    template_centered = template - template_mean[:, None, None]
    candidate_centered = candidate - candidate_mean[:, None, None]
    numerator = float(
        (weight[None] * template_centered * candidate_centered).sum()
    )
    template_energy = float(
        (weight[None] * np.square(template_centered)).sum()
    )
    candidate_energy = float(
        (weight[None] * np.square(candidate_centered)).sum()
    )
    norm = np.sqrt(max(template_energy * candidate_energy, 0.0))
    if norm < 1e-8:
        return float("nan")
    return float(np.clip(numerator / norm, -1.0, 1.0))


def _confidence(score, margin):
    """Map absolute match quality and peak separation to [0, 1]."""
    absolute = np.clip((float(score) - 0.15) / 0.65, 0.0, 1.0)
    unique = np.clip(float(margin) / 0.08, 0.0, 1.0)
    return float(0.75 * absolute + 0.25 * unique)


def _two_link_elbows(base, tip, first_length, second_length):
    offset = np.asarray(tip, dtype=np.float32) - np.asarray(base, dtype=np.float32)
    distance = float(np.linalg.norm(offset))
    minimum = abs(float(first_length) - float(second_length))
    maximum = float(first_length) + float(second_length)
    if not np.isfinite(distance) or distance < max(minimum - 1e-4, 1e-5) or distance > maximum + 1e-4:
        return np.empty((0, 2), dtype=np.float32)
    along = (first_length**2 - second_length**2 + distance**2) / (2.0 * distance)
    height_squared = first_length**2 - along**2
    if height_squared < -1e-3:
        return np.empty((0, 2), dtype=np.float32)
    direction = offset / distance
    midpoint = np.asarray(base, dtype=np.float32) + along * direction
    perpendicular = np.asarray([-direction[1], direction[0]], dtype=np.float32)
    height = np.sqrt(max(float(height_squared), 0.0))
    return np.stack(
        [midpoint + height * perpendicular, midpoint - height * perpendicular]
    ).astype(np.float32)


class EventTriggeredAnchorStateTeacher:
    """Use frozen DINO on events and weighted local ZNCC between events."""

    OUTPUT_KEY = "anchor_observation"
    OUTPUT_DIM = 25
    TEMPLATE_RADIUS = 5
    TEMPLATE_SIZE = 2 * TEMPLATE_RADIUS + 1
    ROLE_TEMPLATE_RADII = (0, 5, 4, 4)
    _EVENT_KEYS = (
        "anchor_event_velocity",
        "anchor_event_key_templates",
        "anchor_event_adaptive_templates",
        "anchor_event_template_weights",
        "anchor_event_template_valid",
        "anchor_event_since_refresh",
    )

    def __init__(self, teacher, config):
        self.teacher = teacher
        self.device = teacher.device
        self.beam_size = teacher.beam_size
        self.image_shape = teacher.image_shape
        self.max_interval = int(_value(config, "max_interval", 12))
        # Four guarded frames cap persistent-failure DINO use at about 20%,
        # below the runtime gate's 25% refresh budget.
        self.min_refresh_gap = int(_value(config, "min_refresh_gap", 4))
        self.confidence_threshold = float(
            _value(config, "confidence_threshold", 0.30)
        )
        self.structure_tolerance = float(
            _value(config, "structure_tolerance", 0.20)
        )
        search_radius = int(_value(config, "color_search_radius", 6))
        self.search_radii = (
            0,
            max(search_radius - 1, 3),
            max(search_radius + 1, 4),
            max(search_radius, 3),
        )
        # Keep the old config name for compatibility.  It is now the minimum
        # accepted absolute ZNCC, not a broad colour-prototype confidence.
        self.min_zncc = float(
            _value(config, "color_confidence_threshold", 0.20)
        )
        self.key_template_weight = 0.70
        self.template_update_alpha = 0.05
        self.template_update_score = 0.65
        self.template_update_margin = 0.025
        if self.max_interval < 1:
            raise ValueError("flat_anchor_event_max_interval must be >= 1")
        if self.min_refresh_gap < 1:
            raise ValueError("flat_anchor_event_min_refresh_gap must be >= 1")
        if min(self.search_radii[1:]) < 1:
            raise ValueError("flat_anchor_event_color_search_radius must be >= 1")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("flat_anchor_event_confidence_threshold must be in [0, 1]")
        if not -1.0 <= self.min_zncc <= 1.0:
            raise ValueError(
                "flat_anchor_event_color_confidence_threshold must be in [-1, 1]"
            )

        self._frames = 0
        self._dino_frames = 0
        self._fast_frames = 0
        self._guarded_frames = 0
        self._fast_attempt_frames = 0
        self._seconds = 0.0
        self._fast_seconds = 0.0
        self._trigger_counts = Counter()
        self._zncc_sum = np.zeros(len(ROLES), dtype=np.float64)
        self._zncc_count = np.zeros(len(ROLES), dtype=np.int64)
        self._zncc_min = np.full(len(ROLES), np.inf, dtype=np.float64)

    def __getattr__(self, key):
        if key == "teacher":
            raise AttributeError(key)
        return getattr(self.teacher, key)

    def initial_state(self, batch, beam_size, device):
        state = self.teacher.initial_state(batch, beam_size, device)
        template_shape = (
            batch, len(ROLES), 3, self.TEMPLATE_SIZE, self.TEMPLATE_SIZE
        )
        weight_shape = (batch, len(ROLES), self.TEMPLATE_SIZE, self.TEMPLATE_SIZE)
        state.update(
            anchor_event_velocity=torch.zeros(
                batch, len(ROLES), 2, dtype=torch.float32, device=device
            ),
            anchor_event_key_templates=torch.zeros(
                template_shape, dtype=torch.float32, device=device
            ),
            anchor_event_adaptive_templates=torch.zeros(
                template_shape, dtype=torch.float32, device=device
            ),
            anchor_event_template_weights=torch.zeros(
                weight_shape, dtype=torch.float32, device=device
            ),
            anchor_event_template_valid=torch.zeros(
                batch, len(ROLES), dtype=torch.bool, device=device
            ),
            anchor_event_since_refresh=torch.full(
                (batch,), self.max_interval, dtype=torch.int64, device=device
            ),
        )
        return state

    @classmethod
    def _role_slice(cls, radius):
        start = cls.TEMPLATE_RADIUS - int(radius)
        stop = cls.TEMPLATE_RADIUS + int(radius) + 1
        return slice(start, stop)

    def _make_template(self, image, point, role_index):
        radius = self.ROLE_TEMPLATE_RADII[role_index]
        template = np.zeros(
            (3, self.TEMPLATE_SIZE, self.TEMPLATE_SIZE), dtype=np.float32
        )
        weight = np.zeros(
            (self.TEMPLATE_SIZE, self.TEMPLATE_SIZE), dtype=np.float32
        )
        if radius == 0:
            return template, weight, False
        patch = _extract_patch(image, point, radius)
        if patch is None or not np.isfinite(patch).all():
            return template, weight, False
        active = self._role_slice(radius)
        template[:, active, active] = patch
        weight[active, active] = _template_weight(patch)
        return template, weight, True

    def _search_candidates(
        self, image, predicted, role_index, key_template,
        adaptive_template, weight, valid, count,
    ):
        if not valid:
            return [], {"template"}
        if not np.isfinite(predicted).all():
            return [], {"nonfinite"}
        radius = self.ROLE_TEMPLATE_RADII[role_index]
        if _extract_patch(image, predicted, radius) is None:
            return [], {"boundary"}
        active = self._role_slice(radius)
        key = key_template[:, active, active]
        adaptive = adaptive_template[:, active, active]
        active_weight = weight[active, active]
        center = np.rint(predicted).astype(np.int64)
        search_radius = self.search_radii[role_index]
        scored = []
        saw_nonfinite = False
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                point = np.asarray(
                    [center[0] + dx, center[1] + dy], dtype=np.float32
                )
                candidate = _extract_patch(image, point, radius)
                if candidate is None:
                    continue
                key_score = _weighted_zncc(key, candidate, active_weight)
                adaptive_score = _weighted_zncc(
                    adaptive, candidate, active_weight
                )
                score = (
                    self.key_template_weight * key_score
                    + (1.0 - self.key_template_weight) * adaptive_score
                )
                if not np.isfinite(score):
                    saw_nonfinite = True
                    continue
                scored.append((float(score), point))
        if not scored:
            return [], {"nonfinite" if saw_nonfinite else "boundary"}
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = []
        nms_radius = max(int(getattr(self.teacher, "nms_radius", 2)), 1)
        for item in scored:
            if all(
                np.linalg.norm(item[1] - previous[1]) > nms_radius
                for previous in selected
            ):
                selected.append(item)
                if len(selected) >= count:
                    break
        return selected, set()

    def _record_scores(self, scores):
        for index in range(1, len(ROLES)):
            score = float(scores[index])
            if np.isfinite(score):
                self._zncc_sum[index] += score
                self._zncc_count[index] += 1
                self._zncc_min[index] = min(self._zncc_min[index], score)

    def _track_one(
        self, image, previous_points, velocity, key_templates,
        adaptive_templates, weights, valid,
    ):
        height, width = self.image_shape
        predicted = previous_points + velocity
        predicted[:, 0] = np.clip(predicted[:, 0], 0, width - 1)
        predicted[:, 1] = np.clip(predicted[:, 1], 0, height - 1)
        predicted[0] = self.teacher.base_anchor

        elbow_candidates, elbow_reasons = self._search_candidates(
            image, predicted[1], 1, key_templates[1],
            adaptive_templates[1], weights[1], valid[1], count=8,
        )
        tip_candidates, tip_reasons = self._search_candidates(
            image, predicted[2], 2, key_templates[2],
            adaptive_templates[2], weights[2], valid[2], count=12,
        )
        goal_candidates, goal_reasons = self._search_candidates(
            image, predicted[3], 3, key_templates[3],
            adaptive_templates[3], weights[3], valid[3], count=6,
        )
        reasons = elbow_reasons | tip_reasons | goal_reasons
        if not elbow_candidates or not tip_candidates or not goal_candidates:
            reasons.add("fallback")
            return None, reasons

        first_length = float(
            self.teacher.geometry["base_to_elbow"]["median_px"]
        )
        second_length = float(
            self.teacher.geometry["elbow_to_control_tip"]["median_px"]
        )
        assignments = []
        for tip_score, tip in tip_candidates:
            solutions = _two_link_elbows(
                predicted[0], tip, first_length, second_length
            )
            for solution in solutions:
                if not (0 <= solution[0] < width and 0 <= solution[1] < height):
                    continue
                for elbow_score, elbow_observation in elbow_candidates:
                    geometry_error = float(
                        np.linalg.norm(elbow_observation - solution)
                    )
                    temporal = (
                        np.linalg.norm(solution - predicted[1])
                        / max(float(self.search_radii[1]), 1.0)
                        + np.linalg.norm(tip - predicted[2])
                        / max(float(self.search_radii[2]), 1.0)
                    )
                    joint_score = (
                        0.45 * elbow_score
                        + 0.55 * tip_score
                        - 0.10 * geometry_error
                        - 0.06 * temporal
                    )
                    assignments.append(
                        (
                            float(joint_score), solution, tip,
                            float(elbow_score), float(tip_score), geometry_error,
                        )
                    )
        assignments.sort(key=lambda item: item[0], reverse=True)
        if not assignments:
            return None, reasons | {"structure", "fallback"}

        best = assignments[0]
        pair_margin = (
            max(best[0] - assignments[1][0], 0.0)
            if len(assignments) > 1 else 0.08
        )
        goal_best = goal_candidates[0]
        goal_margin = (
            max(goal_best[0] - goal_candidates[1][0], 0.0)
            if len(goal_candidates) > 1 else 0.08
        )
        points = predicted.copy()
        points[0] = self.teacher.base_anchor
        points[1] = best[1]
        points[2] = best[2]
        points[3] = goal_best[1]
        scores = np.asarray(
            [1.0, best[3], best[4], goal_best[0]], dtype=np.float32
        )
        confidence = np.asarray(
            [
                1.0,
                _confidence(min(best[3], best[4]), pair_margin),
                _confidence(min(best[3], best[4]), pair_margin),
                _confidence(goal_best[0], goal_margin),
            ],
            dtype=np.float32,
        )
        self._record_scores(scores)

        structure_scale = max(min(first_length, second_length), 1.0)
        if best[5] / structure_scale > self.structure_tolerance:
            reasons.add("structure")
        if float(scores[1:].min()) < self.min_zncc:
            reasons.add("zncc")
        if float(confidence[1:].min()) < self.confidence_threshold:
            reasons.add("low_confidence")
        if not (
            np.isfinite(points).all()
            and np.isfinite(scores).all()
            and np.isfinite(confidence).all()
        ):
            reasons.add("nonfinite")

        updated = adaptive_templates.copy()
        role_scores = (0.0, best[3], best[4], goal_best[0])
        role_margins = (0.0, pair_margin, pair_margin, goal_margin)
        for role_index in range(1, len(ROLES)):
            if (
                role_scores[role_index] < self.template_update_score
                or role_margins[role_index] < self.template_update_margin
            ):
                continue
            radius = self.ROLE_TEMPLATE_RADII[role_index]
            patch = _extract_patch(image, points[role_index], radius)
            if patch is None:
                continue
            active = self._role_slice(radius)
            updated[role_index, :, active, active] = (
                (1.0 - self.template_update_alpha)
                * updated[role_index, :, active, active]
                + self.template_update_alpha * patch
            )
        return (points, confidence, updated), reasons

    def _next_state(
        self, points, previous_points, previous_valid, confidence, fallback,
        velocity, key_templates, adaptive_templates, weights, template_valid,
        since_refresh, output_device,
    ):
        batch = len(points)
        observation = self.teacher._build_observation(
            points, previous_points, previous_valid, confidence, fallback
        )
        tensor = lambda value, dtype=None: torch.as_tensor(
            value, device=output_device, dtype=dtype
        )
        next_state = {
            "anchor_beam_score": torch.full(
                (batch, self.beam_size), -torch.inf,
                dtype=torch.float32, device=output_device,
            ),
            "anchor_beam_tip": torch.zeros(
                batch, self.beam_size, 2, dtype=torch.float32, device=output_device
            ),
            "anchor_beam_elbow": torch.zeros(
                batch, self.beam_size, 2, dtype=torch.float32, device=output_device
            ),
            "anchor_beam_previous_tip": torch.zeros(
                batch, self.beam_size, 2, dtype=torch.float32, device=output_device
            ),
            "anchor_beam_has_velocity": torch.zeros(
                batch, self.beam_size, dtype=torch.bool, device=output_device
            ),
            "anchor_beam_valid": torch.zeros(
                batch, self.beam_size, dtype=torch.bool, device=output_device
            ),
            "anchor_previous_points": tensor(points, torch.float32),
            "anchor_previous_valid": torch.ones(
                batch, dtype=torch.bool, device=output_device
            ),
            self.OUTPUT_KEY: tensor(observation, torch.float32),
            "anchor_event_velocity": tensor(velocity, torch.float32),
            "anchor_event_key_templates": tensor(key_templates, torch.float32),
            "anchor_event_adaptive_templates": tensor(
                adaptive_templates, torch.float32
            ),
            "anchor_event_template_weights": tensor(weights, torch.float32),
            "anchor_event_template_valid": tensor(template_valid, torch.bool),
            "anchor_event_since_refresh": tensor(since_refresh, torch.int64),
        }
        next_state["anchor_beam_score"][:, 0] = 0.0
        next_state["anchor_beam_tip"][:, 0] = next_state["anchor_previous_points"][:, 2]
        next_state["anchor_beam_elbow"][:, 0] = next_state["anchor_previous_points"][:, 1]
        next_state["anchor_beam_previous_tip"][:, 0] = tensor(
            previous_points[:, 2], torch.float32
        )
        next_state["anchor_beam_has_velocity"][:, 0] = True
        next_state["anchor_beam_valid"][:, 0] = True
        return next_state[self.OUTPUT_KEY], next_state

    def _fast_extract(self, image, state, output_device):
        images, leading_shape = self.teacher._to_uint8_images(image)
        batch = len(images)
        if int(np.prod(leading_shape)) != batch:
            raise ValueError(f"Unexpected anchor image leading shape: {leading_shape}")
        previous_points = state["anchor_previous_points"].detach().reshape(
            batch, len(ROLES), 2
        ).cpu().numpy()
        previous_valid = state["anchor_previous_valid"].detach().reshape(
            batch
        ).cpu().numpy()
        velocity = state["anchor_event_velocity"].detach().reshape(
            batch, len(ROLES), 2
        ).cpu().numpy()
        key_templates = state["anchor_event_key_templates"].detach().cpu().numpy()
        adaptive_templates = state[
            "anchor_event_adaptive_templates"
        ].detach().cpu().numpy()
        weights = state["anchor_event_template_weights"].detach().cpu().numpy()
        template_valid = state[
            "anchor_event_template_valid"
        ].detach().cpu().numpy()

        points = np.zeros_like(previous_points)
        confidence = np.zeros((batch, len(ROLES)), dtype=np.float32)
        next_adaptive = adaptive_templates.copy()
        reasons = set()
        for index in range(batch):
            if not previous_valid[index]:
                reasons.add("fallback")
                continue
            tracked, item_reasons = self._track_one(
                images[index], previous_points[index], velocity[index],
                key_templates[index], adaptive_templates[index], weights[index],
                template_valid[index],
            )
            reasons.update(item_reasons)
            if tracked is None:
                continue
            points[index], confidence[index], next_adaptive[index] = tracked
        if reasons:
            return None, None, reasons

        measured_velocity = points - previous_points
        next_velocity = 0.5 * velocity + 0.5 * measured_velocity
        next_velocity[:, 0] = 0.0
        since_refresh = (
            state["anchor_event_since_refresh"].detach().reshape(batch).cpu().numpy()
            + 1
        )
        return (*self._next_state(
            points, previous_points, previous_valid, confidence,
            np.zeros(batch, dtype=bool), next_velocity, key_templates,
            next_adaptive, weights, template_valid, since_refresh, output_device,
        ), reasons)

    def _guarded_fallback(self, state, output_device):
        previous_points = state["anchor_previous_points"].detach().cpu().numpy()
        previous_valid = state["anchor_previous_valid"].detach().cpu().numpy()
        velocity = state["anchor_event_velocity"].detach().cpu().numpy()
        points = previous_points + velocity
        height, width = self.image_shape
        points[..., 0] = np.clip(points[..., 0], 0, width - 1)
        points[..., 1] = np.clip(points[..., 1], 0, height - 1)
        points[:, 0] = self.teacher.base_anchor
        first_length = float(
            self.teacher.geometry["base_to_elbow"]["median_px"]
        )
        second_length = float(
            self.teacher.geometry["elbow_to_control_tip"]["median_px"]
        )
        for index in range(len(points)):
            solutions = _two_link_elbows(
                points[index, 0], points[index, 2], first_length, second_length
            )
            if len(solutions):
                choice = int(
                    np.argmin(np.linalg.norm(
                        solutions - points[index, 1][None], axis=1
                    ))
                )
                points[index, 1] = solutions[choice]
        confidence = np.zeros((len(points), len(ROLES)), dtype=np.float32)
        confidence[:, 0] = 1.0
        since_refresh = (
            state["anchor_event_since_refresh"].detach().cpu().numpy() + 1
        )
        return self._next_state(
            points, previous_points, previous_valid, confidence,
            np.ones(len(points), dtype=bool), 0.5 * velocity,
            state["anchor_event_key_templates"].detach().cpu().numpy(),
            state["anchor_event_adaptive_templates"].detach().cpu().numpy(),
            state["anchor_event_template_weights"].detach().cpu().numpy(),
            state["anchor_event_template_valid"].detach().cpu().numpy(),
            since_refresh, output_device,
        )

    def _full_extract(self, image, state, is_first, output_device, reasons):
        images, _ = self.teacher._to_uint8_images(image)
        previous_points = state["anchor_previous_points"].detach().cpu().numpy()
        previous_valid = state["anchor_previous_valid"].detach().cpu().numpy()
        anchor, next_state = self.teacher.extract(
            image, state, is_first, output_device=output_device
        )
        points = next_state["anchor_previous_points"].detach().cpu().numpy()
        resets = is_first.detach().reshape(len(points), -1).any(dim=-1).cpu().numpy()
        measured_velocity = points - previous_points
        measured_velocity[~previous_valid | resets] = 0.0
        measured_velocity[:, 0] = 0.0

        templates = np.zeros(
            (
                len(points), len(ROLES), 3,
                self.TEMPLATE_SIZE, self.TEMPLATE_SIZE,
            ),
            dtype=np.float32,
        )
        weights = np.zeros(
            (len(points), len(ROLES), self.TEMPLATE_SIZE, self.TEMPLATE_SIZE),
            dtype=np.float32,
        )
        valid = np.zeros((len(points), len(ROLES)), dtype=bool)
        valid[:, 0] = True
        for batch_index in range(len(points)):
            for role_index in range(1, len(ROLES)):
                template, weight, role_valid = self._make_template(
                    images[batch_index], points[batch_index, role_index], role_index
                )
                templates[batch_index, role_index] = template
                weights[batch_index, role_index] = weight
                valid[batch_index, role_index] = role_valid
        next_state.update(
            anchor_event_velocity=torch.as_tensor(
                measured_velocity, dtype=torch.float32, device=output_device
            ),
            anchor_event_key_templates=torch.as_tensor(
                templates, dtype=torch.float32, device=output_device
            ),
            anchor_event_adaptive_templates=torch.as_tensor(
                templates.copy(), dtype=torch.float32, device=output_device
            ),
            anchor_event_template_weights=torch.as_tensor(
                weights, dtype=torch.float32, device=output_device
            ),
            anchor_event_template_valid=torch.as_tensor(
                valid, dtype=torch.bool, device=output_device
            ),
            anchor_event_since_refresh=torch.zeros(
                len(points), dtype=torch.int64, device=output_device
            ),
        )
        frames = len(points)
        self._dino_frames += frames
        for reason in reasons:
            self._trigger_counts[reason] += frames
        return anchor, next_state

    @torch.no_grad()
    def extract(self, image, state, is_first, output_device=None):
        output_device = torch.device(output_device or image.device)
        started = time.perf_counter()
        batch = int(image.reshape(-1, *image.shape[-3:]).shape[0])
        reset = is_first.detach().reshape(batch, -1).any(dim=-1)
        since_refresh = state["anchor_event_since_refresh"].detach().reshape(batch)
        reasons = scheduled_refresh_reasons(
            reset.detach().cpu().numpy(),
            since_refresh.detach().cpu().numpy(),
            self.max_interval,
        )

        if reasons:
            anchor, next_state = self._full_extract(
                image, state, is_first, output_device, reasons
            )
        else:
            fast_started = time.perf_counter()
            self._fast_attempt_frames += batch
            anchor, next_state, reasons = self._fast_extract(
                image, state, output_device
            )
            self._fast_seconds += time.perf_counter() - fast_started
            if reasons:
                if bool((since_refresh >= self.min_refresh_gap).all()):
                    anchor, next_state = self._full_extract(
                        image, state, is_first, output_device, reasons
                    )
                else:
                    guarded_reasons = set(reasons) | {"min_refresh_gap"}
                    for reason in guarded_reasons:
                        self._trigger_counts[reason] += batch
                    anchor, next_state = self._guarded_fallback(
                        state, output_device
                    )
                    self._guarded_frames += batch
            else:
                self._fast_frames += batch

        self._frames += batch
        self._seconds += time.perf_counter() - started
        return anchor, next_state

    def metrics(self):
        frames = max(self._frames, 1)
        fast_attempts = max(self._fast_attempt_frames, 1)
        metrics = {
            "frames": float(self._frames),
            "dino_frames": float(self._dino_frames),
            "fast_frames": float(self._fast_frames),
            "guarded_frames": float(self._guarded_frames),
            "fast_attempt_frames": float(self._fast_attempt_frames),
            "refresh_fraction": float(self._dino_frames / frames),
            "fallback_fraction": float(
                self.teacher.metrics().get("fallback_fraction", 0.0)
            ),
            "ms_per_frame": float(1000.0 * self._seconds / frames),
            "fast_ms_per_attempt": float(
                1000.0 * self._fast_seconds / fast_attempts
            ),
        }
        trigger_reasons = (
            "reset", "max_interval", "zncc", "low_confidence", "boundary",
            "nonfinite", "template", "fallback", "structure",
            "min_refresh_gap",
        )
        metrics.update(
            {
                f"trigger_{reason}": float(self._trigger_counts.get(reason, 0))
                for reason in trigger_reasons
            }
        )
        for index, role in enumerate(ROLES[1:], start=1):
            count = int(self._zncc_count[index])
            metrics[f"zncc_{role}_mean"] = float(
                self._zncc_sum[index] / max(count, 1)
            )
            metrics[f"zncc_{role}_min"] = float(
                self._zncc_min[index] if count else 0.0
            )
        return metrics
