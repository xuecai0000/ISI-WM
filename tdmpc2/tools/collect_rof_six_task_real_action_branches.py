"""Collect strict real-action siblings for the six-task control screen.

This is a new, deliberately small entry point around the already-audited
``collect_rof_real_action_branches`` implementation.  The legacy two-task
diagnostic and its CLI are not changed.  Every other invariant (fresh-reset
prefix replay, genuinely executed sibling actions, indivisible root splits,
and oracle/model-input namespace separation) remains owned and validated by
that implementation.

The output is still a *diagnostic* dataset.  It must be converted by
``materialize_rof_real_sibling_training_capsule`` before controller training;
the controller runner refuses the diagnostic manifest directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tdmpc2.tools import collect_rof_real_action_branches as base


TASKS = (
	'finger-spin',
	'cartpole-swingup',
	'reacher-easy',
	'cup-catch',
	'walker-walk',
	'acrobot-swingup',
)


def parse_args(argv=None):
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument('--task', choices=TASKS, required=True)
	parser.add_argument('--runtime-config', type=Path, required=True)
	parser.add_argument('--checkpoint', type=Path, required=True)
	parser.add_argument('--output-root', type=Path, required=True)
	parser.add_argument('--roots', type=int, default=60)
	parser.add_argument('--train-roots', type=int, default=40)
	parser.add_argument('--validation-roots', type=int, default=10)
	parser.add_argument('--anchor-steps', default='80,160,240,320,400')
	parser.add_argument('--branch-magnitude', type=float, default=0.8)
	parser.add_argument('--env-seed-base', type=int, default=4242430)
	parser.add_argument('--background-seed-base', type=int, default=16180340)
	parser.add_argument('--planner-seed-base', type=int, default=86754000)
	parser.add_argument('--continuation-seed-base', type=int, default=27182810)
	return parser.parse_args(argv)


def main(argv=None):
	args = parse_args(argv)
	# The audited collector resolves this allow-list in its own module.  Change
	# it only inside this dedicated process; importing/running the legacy entry
	# point in another process retains its original two-task contract.
	base.TASKS = TASKS
	manifest = base.collect(args)
	print(
		'ROF_SIX_TASK_REAL_ACTION_BRANCH_DATASET_COMPLETE',
		json.dumps({
			'task': manifest['task'],
			'roots': len(manifest['groups']),
			'manifest': str(
				(args.output_root / base.MANIFEST_NAME).resolve()
			),
		}, allow_nan=False),
		flush=True,
	)
	return 0


if __name__ == '__main__':
	raise SystemExit(main())
