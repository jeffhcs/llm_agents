
from llm import prompt_llm, wrap_message
import alfworld.agents.environment as environment
import yaml
config_path = 'base_config.yaml'

# load config
with open(config_path, 'r') as reader:
    config = yaml.safe_load(reader)

env_type = config['env']['type'] # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'

import re

########



def run_trajectory(policy, game_id = 0, gpt='gpt-3.5-turbo', max_steps=16):

    def get_prompt_react(trajectory, policy, few_shot = False):
        def get_policy_message(policy):
            prompt = f"""
Here is a guide for completing the next task:
{policy}

Please follow the above guide carefully in your next game. 
You should often explicitly refer to the guide in your thoughts.
"""
            return wrap_message('system', prompt)

        action_commands = """
    >>> go to [object]
    >>> take [object] from [recep]
    >>> put [object] in/on [recep]
    >>> open [recep]
    >>> close [recep]
    >>> toggle [object/recep]
    >>> clean [object] with [recep]
    >>> heat [object] with [recep]
    >>> cool [object] with [recep]
    >>> inventory
"""

        instructions = f"""
You are playing a text based game.

The available action commands are:
{action_commands}
You must follow the command format exactly. To use the put command, make sure you type `in/on` instead of just `in` or just `on`. Do not try any commands that are not listed.

At each step, provide your thinking and your next action.
All of your messages should strictly follow the format below:
think: [your thoughts]
>>> [your action]

Do not produce any additional output.
Ensure that you provide exactly one command in each message.
"""
        prompt = [wrap_message('system', instructions)]
        # if few_shot:
            # prompt += [get_few_shot()]
        if policy:
            prompt += [get_policy_message(policy)]
        prompt += trajectory

        return prompt

    def parse_command(response):
        command_match = re.search(r'>>> (.+)', response)
        assert command_match, "No command found in LLM response."
        command = command_match.group(1)

        return command

    def step_env(command):
        obs, reward, done, info = env.step([command])
        return obs[0], done[0]

    def step_react(trajectory):

        # get command from llm
        prompt = get_prompt_react(trajectory, policy)
        response = prompt_llm(prompt, gpt)
        trajectory.append(wrap_message('assistant', response))
        command = parse_command(response)

        obs, done = step_env(command)
        trajectory.append(wrap_message('user', obs))

        return done
    
    def init_env(game_id):
        env = getattr(environment, env_type)(config, train_eval='eval_out_of_distribution')
        env = env.init_env(batch_size=1)
        env.skip(game_id)
        obs, info = env.reset()
        return env, obs[0]

    env, init_obs = init_env(game_id)
    trajectory = [wrap_message('user', init_obs)]
    success = False

    for i in range(max_steps):
        done = step_react(trajectory)
        if done:
            success = True
            trajectory.append(wrap_message('user', 'TASK COMPLETED!'))
            break
    else:
        trajectory.append(wrap_message('user', 'Game terminated due to reaching max steps'))
        trajectory.append(wrap_message('user', 'TASK FAILED!'))
    
    return trajectory, success


def refine_policy(policy, trajectories, gpt='gpt-3.5-turbo'):
    def extract_policy(response):
        match = re.search(r'Guide: ([\s|\S]+)', response)
        assert match, "No guide found in LLM response."
        return match.group(1)

    prompt = f"""
```
{trajectories}
```
"""
    if policy:
        prompt += f"""
In the above games, you have been following this guide:
```
{policy}
```
Please refine this guide based on the insights gained from the games.
"""
    else:
        prompt += f"""
In the above games, you have been trying to complete a particular task.
Develop a guide based on the insights gained from the games for completing this particular task.
"""
    prompt += f"""
If there are any failures, reflect on what went wrong and what you could've done differently to avoid this error.
Revise the procedure based on the insights gained from the trajectory.
First brainstorm some thoughts and then output your guide.
The guide will be used to help future players in possible different situations, so don't make it too specific to a particular setting.
You can describe what to do in a variety of situations.

Thoughts: <your thoughts>
Guide: <your guide>

Do not produce any additional output.
    """
    response = prompt_llm(prompt, gpt)
    new_policy = extract_policy(response)
    return new_policy


def main(num_iterations = 3, batch_size = 3):
    policy = None

    policies_history = []
    trajectories = []

    for i in range(num_iterations):
        batch = []
        for j in range(batch_size):
            trajectory, success = run_trajectory(policy)
            batch.extend(trajectory)
            trajectories.append((trajectory, success))
        
        policy = refine_policy(policy, batch)
        policies_history.append(policy)

