import os
import json
import sys

def main():
    with open('../goal.json') as f:
        data = json.load(f)

    mp = None
    if sys.argv[1:]:
        mp = int(sys.argv[1])

    for mn, task in data.items():
        if mp is not None and mp != int(mn[3]): continue    
        seed = task['seed'][0]
        start = task['start']
        start = ','.join([str(x) for x in start])
        goal = task['goal']
        goal = ','.join([str(x) for x in goal])

        os.system(f'python run_control.py -m {mn} -s {seed} -st {start} -gt {goal}' )   
    
if __name__ == '__main__':
    main()