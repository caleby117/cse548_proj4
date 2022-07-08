import subprocess
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--scenario', type=int, action='append')

clargs = parser.parse_args(sys.argv[1:])

class Scenario:
    def __init__(self, name, directory, train, test):
        self.name = name
        self.dir = directory
        self.train = train
        self.test = test
        self.cmd = f'python3 fnn_sample.py --traindata {self.dir}{self.train} '\
            f'--testdata {self.dir}{self.test} -s {self.name} -e 5'
    
    def train_n_test(self):
        # Execute the training and testing of the CNN for this scenario
        # with a subprocess
        print(self.cmd)
        print(f'Training and testing FNN for scenario {self.name}. This should take about 10 minutes')
        with subprocess.Popen(self.cmd.split(), stdout=subprocess.PIPE) as proc:
            for c in iter(lambda: proc.stdout.read(1), b''):
                sys.stdout.write(c.decode())

        print('Done')

def main():
    scenario1 = Scenario('1','./scenario1/', \
        'Training-a1-a3_standardized.csv', 'Testing-a2-a4_standardized.csv')
    scenario2 = Scenario('2','./scenario2/', \
        'Training-a1-a2_standardized.csv', 'Testing-a1_standardized.csv')
    scenario3 = Scenario('3','./scenario3/', \
        'Training-a1-a2_standardized.csv', 'Testing-a1-a2-a3_standardized.csv')
    
    all_scenarios = [scenario1, scenario2, scenario3]
    
    scenarios_to_train = list(map(lambda x: int(x)-1, sorted(clargs.scenario)))
    print(scenarios_to_train)
    for i in scenarios_to_train:
        all_scenarios[i].train_n_test()

if __name__ == '__main__': 
    main()