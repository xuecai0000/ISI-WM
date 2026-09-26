"""Dependency-light contract for native-resolution evaluator wrapper lookup.

The environment factory imports its wrappers through the top-level ``envs``
namespace.  This check prevents the evaluator from importing the same source
through ``tdmpc2.envs`` and then performing ``isinstance`` with a distinct
class object.  It requires strict identity-based traversal: the real factory
class is found, while a same-named alias and a genuinely missing wrapper both
fail closed.

Run from the repository root::

    python tdmpc2/check_cutie_native_resolution_wrapper_contract.py
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


EVALUATOR = (
	Path(__file__).resolve().parent
	/ 'tools'
	/ 'evaluate_cutie_native_resolution_tracker.py'
)
SOURCE = EVALUATOR.read_text(encoding='utf-8')
TREE = ast.parse(SOURCE, filename=str(EVALUATOR))


def _function_node(name: str) -> ast.FunctionDef:
	for node in TREE.body:
		if isinstance(node, ast.FunctionDef) and node.name == name:
			return node
	raise AssertionError(f'Evaluator function {name!r} is missing.')


def _load_find_wrapper():
	"""Compile only the stdlib-only traversal, avoiding DMC/Cutie imports."""
	node = _function_node('_find_wrapper')
	module = ast.Module(body=[node], type_ignores=[])
	namespace: dict[str, object] = {}
	exec(compile(ast.fix_missing_locations(module), str(EVALUATOR), 'exec'), namespace)
	return namespace['_find_wrapper']


_find_wrapper = _load_find_wrapper()


class _Layer:
	def __init__(self, env=None):
		self.env = env


class NativeResolutionWrapperLookupContract(unittest.TestCase):
	def test_evaluator_uses_factory_module_for_both_env_and_wrapper_type(self):
		evaluate = _function_node('evaluate')
		prepare = _function_node('_prepare_tracking_environment')
		imports_factory_module = any(
			isinstance(node, ast.Import)
			and any(
				alias.name == 'envs.dmcontrol'
				and alias.asname == 'dmcontrol_env'
				for alias in node.names
			)
			for node in ast.walk(evaluate)
		)
		self.assertTrue(
			imports_factory_module,
			'Evaluator must bind the exact top-level env factory module.',
		)

		calls = [node for node in ast.walk(prepare) if isinstance(node, ast.Call)]
		make_env_calls = [
			node for node in calls
			if isinstance(node.func, ast.Attribute)
			and isinstance(node.func.value, ast.Name)
			and node.func.value.id == 'dmcontrol_env'
			and node.func.attr == 'make_env'
		]
		self.assertEqual(len(make_env_calls), 1)

		lookups = [
			node for node in calls
			if isinstance(node.func, ast.Name)
			and node.func.id == '_find_wrapper'
		]
		self.assertEqual(len(lookups), 1)
		self.assertEqual(len(lookups[0].args), 2)
		wrapper_type = lookups[0].args[1]
		self.assertIsInstance(wrapper_type, ast.Attribute)
		self.assertIsInstance(wrapper_type.value, ast.Name)
		self.assertEqual(wrapper_type.value.id, 'dmcontrol_env')
		self.assertEqual(wrapper_type.attr, 'ColorMultiVideoBackgroundWrapper')

		self.assertEqual(
			sum(
				isinstance(node, ast.Call)
				and isinstance(node.func, ast.Name)
				and node.func.id == '_prepare_tracking_environment'
				for node in ast.walk(evaluate)
			),
			1,
		)

		for node in ast.walk(TREE):
			if isinstance(node, ast.ImportFrom):
				self.assertNotEqual(
					node.module, 'tdmpc2.envs.wrappers.video_background'
				)

	def test_strict_traversal_finds_the_factory_class_instance(self):
		actual_type = type('ColorMultiVideoBackgroundWrapper', (), {})
		factory_module = type(
			'FactoryModule', (),
			{'ColorMultiVideoBackgroundWrapper': actual_type},
		)
		actual = actual_type()
		env = _Layer(_Layer(actual))
		self.assertIs(
			_find_wrapper(
				env, factory_module.ColorMultiVideoBackgroundWrapper
			),
			actual,
		)

	def test_same_named_duplicate_module_class_is_not_accepted(self):
		actual_type = type('ColorMultiVideoBackgroundWrapper', (), {})
		duplicate_type = type('ColorMultiVideoBackgroundWrapper', (), {})
		env = _Layer(actual_type())
		with self.assertRaisesRegex(
			RuntimeError,
			'^ColorMultiVideoBackgroundWrapper is missing from the environment chain\\.$',
		):
			_find_wrapper(env, duplicate_type)

	def test_missing_wrapper_and_cyclic_chain_fail_closed(self):
		wrapper_type = type('ColorMultiVideoBackgroundWrapper', (), {})
		with self.assertRaisesRegex(RuntimeError, 'is missing'):
			_find_wrapper(_Layer(_Layer()), wrapper_type)

		left, right = _Layer(), _Layer()
		left.env, right.env = right, left
		with self.assertRaisesRegex(RuntimeError, 'is missing'):
			_find_wrapper(left, wrapper_type)

	def test_environment_setup_failure_closes_the_environment(self):
		node = _function_node('_prepare_tracking_environment')
		module = ast.Module(body=[node], type_ignores=[])
		closed = []

		class _Env:
			def close(self):
				closed.append(True)

		class _Factory:
			ColorMultiVideoBackgroundWrapper = type('Background', (), {})

			@staticmethod
			def make_env(config):
				return _Env()

		namespace = {
			'_Config': SimpleNamespace,
			'_find_wrapper': lambda env, kind: (_ for _ in ()).throw(
				RuntimeError('setup failed')
			),
			'_find_physics': lambda env: None,
			'SPLIT': 'validation',
		}
		exec(
			compile(ast.fix_missing_locations(module), str(EVALUATOR), 'exec'),
			namespace,
		)
		args = SimpleNamespace(
			task='acrobot-swingup', env_seed=1,
			video_root=Path('.'), manifest_dir=None,
			background_total_frames=1000, background_cache_size=8,
			background_seed=2,
		)
		with self.assertRaisesRegex(RuntimeError, 'setup failed'):
			namespace['_prepare_tracking_environment'](
				args, ('upper_arm', 'lower_arm'), _Factory, {}, None, None
			)
		self.assertEqual(closed, [True])


if __name__ == '__main__':
	unittest.main(verbosity=2)
