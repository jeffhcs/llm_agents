

from llm import prompt_llm
import alfworld.agents.environment as environment
import yaml
config_path = 'base_config.yaml'

# load config
with open(config_path, 'r') as reader:
    config = yaml.safe_load(reader)

env_type = config['env']['type'] # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'

import re

########


example_skills = [
    {'description': 'find and pick up object',
     'policy': """
#### Procedure:
1. Identify the object you need to find.
2. Think about probable locations where the object might be.
3. Move to the first probable location.
4. Check if the object is there.
   - If not, repeat steps 3 and 4 for the next probable locations until the object is found.
5. Once the object is located, execute the command to pick it up.
"""},
    {'description': 'heat object in inventory with microwave',
     'policy': """
#### Procedure:
1. Identify the object you need to heat.
2. Verify that the object is currently in the inventory.
3. Move to any microwave.
4. Execute the command to heat the object with the microwave. You do not need to first open the microwave.
"""}
]

def wrap_message(role, content):
    return {
        'role': role,
        'content': content
    }


def run_trajectory(skills, game_id = 0, gpt='gpt-3.5-turbo', max_steps=10):

    def get_prompt_react(trajectory, skills):
        def get_skills_message(skills):
            skills_list = "\n   ".join([f"{i+1}. {skill['description']}" for i, skill in enumerate(skills)])
            skills_text = f"""
For the following game, the available skills are:
{skills_list}
"""
            return wrap_message('system', skills_text)

        def get_few_shot():
            example_skills_1 = [
                {'description': 'place an object from inventory in/on a receptacle',
                 'policy': """
1. Ensure that the object is in the inventory using the inventory command. 
2. Go to the receptacle where you want to place the object.
3. Check if the receptacle is closed. If so, open it.
4. Execute the command to place the object in/on the receptacle.
"""},
            ]
            trajectory_text = f"""
-= Welcome to TextWorld, ALFRED! =-
You are in the middle of a room. Looking quickly around you, you see a armchair 1, a coffeetable 1, a diningtable 1, a drawer 7, a drawer 6, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a garbagecan 1, a ottoman 1, a sidetable 1, and a sofa 1.
Your task is to: put a newspaper in ottoman.

think: My goal is to put a newspaper in the ottoman. There is a skill to place and object from inventory, but I must first obtain the object. To do that, I first need to find the newspaper. I will begin by searching places where a newspaper might be found.
>>> go to coffeetable 1

You arrive at loc 6. On the coffeetable 1, you see a book 1, a laptop 1, a pencil 1, a tissuebox 1, and a vase 1.

think: I did not find the newspaper on the coffeetable. I will now search the armchair.
>>> go to armchair 1

You arrive at loc 4. On the armchair 1, you see nothing.

think: I did not find the newspaper on the armchair. I will now search the diningtable.
>>> go to diningtable 1

You arrive at loc 17. On the diningtable 1, you see a bowl 2, a bowl 1, a newspaper 1, a plate 1, and a vase 2.

think: I found the newspaper. Now I need to pick it up.
>>> take newspaper 1 from diningtable 1

You pick up the newspaper 1 from the diningtable 1.

think: I have the newspaper. Now I can use the skill to place an object from inventory in/on a receptacle. 
>>> recall 1: place an object from inventory in/on a receptacle

You recall a guide for the skill "{example_skills_1[0]['description']}".
```
{example_skills_1[0]['policy']}
```

think: According to the skill guide, I should first verify that the object is in the inventory. I will do that now.
>>> inventory

You are carrying the following objects: newspaper 1.

think: I have the newspaper in my inventory. Now I need to go to the ottoman to place the newspaper.
> go to ottoman 1

You arrive at loc 25. On the ottoman 1, you see a keychain 1, and a remotecontrol 1.

think: I am at the ottoman. The ottoman does not seem to be closed so I do not need to open it. I can now place the newspaper in/on the ottoman.
> put newspaper 1 in/on ottoman 1

You put the newspaper 1 in/on the ottoman 1.
TASK COMPLETED!
"""
            trajectory_messages = [wrap_message('assistant' if i % 2 else 'user', message) for i, message in enumerate(trajectory_text.split('\n\n'))]
            few_shot_1 = [get_skills_message(example_skills_1)] + trajectory_messages
            return few_shot_1

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

You additionally have access to the special command `recall` to recall instructions about a specific skill.
You can also recall a skill by typing ">>> recall [skill_id]: [skill description]".
The set of available skills are different for each game, and will be provided at the start of the game.
Make sure to only recall the skills available for your current game.
When what you're trying to do is similar to an available skill, you should recall it instead of trying to figure out the steps yourself.

At each step, provide your thinking and your next action.
Provide all your thoughts and actions in the format:
think: [your thoughts]
>>> [your action]

"""
        prompt = [wrap_message('system', instructions)] + get_few_shot() + [get_skills_message(skills)] + trajectory
        return prompt

    def parse_command(response):
        command_match = re.search(r'>>> (.+)', response)
        assert command_match, "No command found in LLM response."
        command = command_match.group(1)

        recall_match = re.search(r'recall (\d+)', command)
        if recall_match:
            skill_id = int(recall_match.group(1))
            return True, skill_id
        else:
            return False, command

    def recall(skill_id):
        skill = skills[skill_id]
        recalled_message = f"""
You recall a guide for the skill "{skill['name']}".
```
{skill['procedure']}
```
"""
        return recalled_message
        

    def step_env(command):
        obs, reward, done, info = env.step([command])
        return obs[0], done

    def step_react(trajectory):

        # get command from llm
        prompt = get_prompt_react(trajectory, skills)
        response = prompt_llm(prompt, gpt)
        trajectory.append(wrap_message('assistant', response))
        is_recall, command = parse_command(response)

        if is_recall: # llm is recalling a skill

            recalled_message = recall(command)
            trajectory.append(wrap_message('user', recalled_message))
            done = False

        else: # llm is executing an action

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
        # trajectory terminated due to reaching max steps
        trajectory.append(wrap_message('user', 'Game terminated due to reaching max steps'))
        trajectory.append(wrap_message('user', 'TASK FAILED!'))
    
    return trajectory, success



def tag_trajectory(trajectory, tags, gpt='gpt-3.5-turbo'):

    def parse_tags(response):
        return response

    instructions = f"""
Suggest some skills that were demonstrated in the above game.
Any suggested skills must be general and reusable, not involving any specific object.
The skill should be non-trivial, requiring at least two steps to complete.
Some examples of skills are "clean object in inventory" or "find and pick up object".
For each suggested skill, first describe how it was demonstrated before naming the skill.
Your output must be formatted as follows:

Observation: [description of the skill as demonstrated in the game]
Skill: [name of the skill]
"""

    if tags:
        instructions += f"""
Here are some more examples of skills:
{"  -" + "\n  -".join(tags)}

If any of the skills you suggest are similar to any above listed skills, you must use the exact same wording as the example skill.
"""

    prompt = trajectory + [wrap_message('system', instructions)]
    response = prompt_llm(prompt, gpt)
    tags = parse_tags(response)
    return tags


def distill_trajectories(trajectories, tag, gpt='gpt-3.5-turbo'):

    def parse_skills(response):
        match = re.search(r'Guide:\s([\s|\S]+)', response)
        assert match, "No guide found in LLM response."
        return match.group(1)
    
    instructions = f"""
The above were games demonstrating the skill "{tag}".

Use the games to learn about the skill "{tag}" to create a general guide for the skill.
The guide will be used to help future players in different games, so it should be general and not reference anything too specific.

First reflect on how the skill was used in the games, and then provide a general guide for the skill.

Format your output as follows:

Skill: {tag}
Reflection: [your reflection]
Guide: [general guide for the skill]

Do not produce any additional output.
"""
    prompt = trajectories + [wrap_message('system', instructions)]
    response = prompt_llm(prompt, gpt)
    skill = parse_skills(response)
    return skill


def main(max_iterations = 5, distill_batch_size = 3):
    skills = []
    tagged_trajectories = {}

    for i in range(max_iterations):
        trajectory = run_trajectory(skills)
        tags = tag_trajectory(trajectory, tagged_trajectories.keys())
        
        for t in tags:
            tagged_trajectories.setdefault(t, []).append(trajectory)

            if len(tagged_trajectories[t]) == distill_batch_size:
                new_skills = distill_trajectories(tagged_trajectories[t], t)
                skills.extend(new_skills)
    





