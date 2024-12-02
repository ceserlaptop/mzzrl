from alg.maddpg.utils import get_trainers
import torch
import torch.nn as nn


class agents:
    def __init__(self, env, obs_shape_n, action_shape_n, arglist):
        self.actors_cur = None
        self.critics_cur = None
        self.actors_tar = None
        self.critics_tar = None
        self.optimizers_a = None
        self.optimizers_c = None
        self.actors_cur, self.critics_cur, self.actors_tar, self.critics_tar, self.optimizers_a, self.optimizers_c = \
            get_trainers(env, obs_shape_n, action_shape_n, arglist)
        print("maddpg Agents inited!")

    def agents_train(self, arglist, game_step, update_cnt, memory, obs_size, action_size):
        """
        use this func to make the "main" func clean
        par:
        |input: the data for training
        |output: the data for next update
        """
        # update all trainers, if not in display or benchmark mode
        if game_step > arglist.learning_start_step and \
                (game_step - arglist.learning_start_step) % arglist.learning_fre == 0:
            if update_cnt == 0:
                print('Start training ...')
            # update the target par using the cur
            update_cnt += 1

            # update every agent in different memory batch
            for agent_idx, (actor_c, actor_t, critic_c, critic_t, opt_a, opt_c) in \
                    enumerate(zip(self.actors_cur, self.actors_tar, self.critics_cur, self.critics_tar,
                                  self.optimizers_a, self.optimizers_c)):
                # sample the experience
                _obs_n, _action_n, _rew_n, _obs_n_t, _done_n = memory.sample(
                    arglist.batch_size, agent_idx)  # Note_The func is not the same as others

                # --use the data to update the CRITIC
                # set the data to gpu
                rew = torch.tensor(_rew_n, dtype=torch.float, device=arglist.device)
                done_n = torch.tensor(~_done_n, dtype=torch.float, device=arglist.device)
                action_n = torch.from_numpy(_action_n).to(arglist.device, torch.float)
                obs_n = torch.from_numpy(_obs_n).to(arglist.device, torch.float)
                obs_n_t = torch.from_numpy(_obs_n_t).to(arglist.device, torch.float)
                # cal the loss
                action_tar = torch.cat([a_t(obs_n_t[:, obs_size[idx][0]:obs_size[idx][1]]).detach()
                                        for idx, a_t in enumerate(self.actors_tar)],
                                       dim=1)  # get the action in next state
                q = critic_c(obs_n, action_n).reshape(-1)  # q value in current state
                q_t = critic_t(obs_n_t, action_tar).reshape(-1)  # q value in next state
                tar_value = q_t * arglist.gamma * done_n + rew  # q_*gamma*done + reward
                loss_c = torch.nn.MSELoss()(q, tar_value)  # bellman equation
                # update the parameters
                opt_c.zero_grad()
                loss_c.backward()
                nn.utils.clip_grad_norm_(critic_c.parameters(), arglist.max_grad_norm)
                opt_c.step()

                # --use the data to update the ACTOR
                # There is no need to cal other agent's action
                model_out, policy_c_new = actor_c(
                    obs_n[:, obs_size[agent_idx][0]:obs_size[agent_idx][1]], model_original_out=True)
                # update the action of this agent
                action_n[:, action_size[agent_idx][0]:action_size[agent_idx][1]] = policy_c_new
                loss_pse = torch.mean(torch.pow(model_out, 2))
                loss_a = torch.mul(-1, torch.mean(critic_c(obs_n, action_n)))

                opt_a.zero_grad()
                (1e-3 * loss_pse + loss_a).backward()
                nn.utils.clip_grad_norm_(actor_c.parameters(), arglist.max_grad_norm)
                opt_a.step()

            # update the target network
            actors_tar = self.update_train_tar(self.actors_cur, self.actors_tar, arglist.tau)
            critics_tar = self.update_train_tar(self.critics_cur, self.critics_tar, arglist.tau)

        return update_cnt, self.actors_cur, self.actors_tar, self.critics_cur, self.critics_tar

    def update_train_tar(self, agents_cur, agents_tar, tau):
        """
        update the trainers_tar par using the trainers_cur
        This way is not the same as copy_, but the result is the same
        out:
        |agents_tar: the agents with new par updated towards agents_current
        """
        for agent_c, agent_t in zip(agents_cur, agents_tar):
            key_list = list(agent_c.state_dict().keys())
            state_dict_t = agent_t.state_dict()
            state_dict_c = agent_c.state_dict()
            for key in key_list:
                state_dict_t[key] = state_dict_c[key] * tau + \
                                    (1 - tau) * state_dict_t[key]
            agent_t.load_state_dict(state_dict_t)
        return agents_tar
