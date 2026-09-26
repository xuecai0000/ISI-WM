"""Materialize a six-screen real-sibling dataset without widening legacy CLIs."""

from __future__ import annotations

from tdmpc2.tools import collect_rof_real_action_branches as collector
from tdmpc2.tools import materialize_rof_real_sibling_training_capsule as base
from tdmpc2.tools.collect_rof_six_task_real_action_branches import TASKS


def main(argv=None):
	# ``base.materialize`` deliberately re-runs the producing collector's strict
	# validator.  Extend only that validator's allow-list in this dedicated
	# process; importing either legacy entry point elsewhere remains unchanged.
	collector.TASKS = TASKS
	base.collector.TASKS = TASKS
	if set(base.collector.TASKS) != set(TASKS):
		raise RuntimeError('Six-task materializer validator allow-list was not installed.')
	return base.main(argv)


if __name__ == '__main__':
	raise SystemExit(main())
