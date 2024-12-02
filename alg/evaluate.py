import torch
import torch.nn.functional as F
import numpy as np


def evaluate(env, arglist, actors_cur):
    # 创建环境
    obs_n = env.reset()

    episode_coverage = []
    episode_outRange = [0]
    episode_collision = [0]
    episode_rewards = [0]
    done_steps = []

    for test_episode in range(arglist.evaluate_episode_num):
        episode_step = 0
        while True:
            # get action
            action_n = []
            for actor, obs in zip(actors_cur, obs_n):
                model_out, _ = actor(torch.from_numpy(obs).to(arglist.device, torch.float), model_original_out=True)
                action_n.append(F.softmax(model_out, dim=-1).detach().cpu().numpy())

            # interact with env
            new_obs_n, rew_n, done_n = env.step(action_n)
            obs_n = new_obs_n

            if env.world.collision:
                episode_collision[-1] += 1
            if env.world.outRange:
                episode_outRange[-1] += 1

            episode_step += 1
            terminal = (episode_step >= arglist.max_step_len) or all(done_n)
            if terminal:
                episode_outRange[-1] /= episode_step
                episode_outRange.append(0)
                episode_collision[-1] /= episode_step
                episode_collision.append(0)
                cov = env.world.coverage_rate
                episode_coverage.append(cov)
                episode_rewards.append(np.sum(rew_n))
                if env.world.coverage_rate == 1.0:
                    done_steps.append(episode_step)
                else:
                    done_steps.append(arglist.max_step_len)
                obs_n = env.reset()
                break

    return episode_coverage, episode_rewards[:-1], episode_collision[:-1], episode_outRange[-1], done_steps
