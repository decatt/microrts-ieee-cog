run main.py to test
python main.py --map_path maps/16x16/basesWorkers16x16A.xml --op_ai coacAI

add opponent_ai.jar in gym_microrts\microrts
and add code

def oppent_ai(utt):
    from ai import opponent_ai
    return opponent_ai()

in gym_microrts\microrts_ai.py