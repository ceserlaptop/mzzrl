import env.cpp_env.scenarios as scenarios
from env.cpp_env.environment import MultiAgentEnv


def make_env(arglist):
    """
    create the environment from script
    """
    scenario_name = arglist.scenario_name
    scenario = scenarios.load(scenario_name + ".py").Scenario(r_cover=0.1, r_comm=0.4, comm_r_scale=0.9,
                                                              comm_force_scale=5.0, env_size=10, arglist=arglist)
    world = scenario.make_world()
    if arglist.benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,
                            done_callback=scenario.done)
        # env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


