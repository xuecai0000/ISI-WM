"""Print MuJoCo geometry/body catalogs for object-role configuration."""

from dm_control import suite


for domain, task in (('cheetah', 'run'), ('quadruped', 'run'), ('quadruped', 'walk')):
	env = suite.load(domain, task)
	model = env.physics.model
	print('TASK', domain, task)
	print('GEOMS', [model.id2name(i, 'geom') for i in range(model.ngeom)])
	print('BODIES', [model.id2name(i, 'body') for i in range(model.nbody)])
