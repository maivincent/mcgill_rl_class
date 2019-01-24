from environment import Environment
from algorithms import ActionEliminationAlgo
import matplotlib.pyplot as plt
import csv
import os


FIXED_NB_STEPS = 7000
NB_RUNS = 5000
EXP_NAME = "action_elimination_article"

class MainLoop():
    def __init__(self):
        self.delta = 0.1
        self.epsilon = 0.01
        self.algo = ActionEliminationAlgo(self.delta, self.epsilon) # Change here for another algorithm
        self.env = Environment()
        self.action_memory = []
        self.reward_memory = []
        self.step = 0

    def doOneStep(self):
        self.step += 1

        action = self.algo.nextAction()
        self.action_memory.append(action)

        reward = self.env.draw(action)
        self.reward_memory.append(action)

        self.algo.update(action,reward)

    def findBestArm(self):
        # HERE: Choose the stop condition: algorithm has solved problem OR fixed number of steps (better to do averages per step)
        #while not self.algo.isDone():
        while self.step < FIXED_NB_STEPS:
            self.doOneStep()
        return self.algo.result()

    def get_action_memory(self):
        return self.action_memory


class Drawer():
    def __init__(self, exp_name):
        self.output_path_root = "./experiments/" + exp_name
        self.make_dir("./experiments")
        self.make_dir(self.output_path_root)

    def save_png(self, x, sum_action_step, x_label, y_label, plot_title):
        plt.subplots()
        for i in range(6):
            y = sum_action_step[i]
            plt.plot(x, y)
        plt.title(plot_title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        output_path = self.output_path_root + "/" + plot_title + ".png"
        plt.savefig(output_path, bbox_inches="tight")

    def save_csv(self, sum_action_step, csv_title):
        #path = self.output_path_root + "/" + csv_title + ".csv"
        #if not os.path.exists(path):
        #    with open(path, "w"):
        #        pass
        #scores_file = open(path, "a")
        #for action in sum_action_step:
        #    with scores_file:
        #        writer = csv.writer(scores_file)
        #        writer.writerow(action)
        pass

    def make_dir(self, path):
        # Check if directory exists, if not, create it
        if not os.path.exists(path):
            try:  
                os.mkdir(path)
            except OSError:  
                print ("Creation of the directory %s failed" % path)
            else:  
                print ("Successfully created the directory %s " % path)        






if __name__ == "__main__":
    # Initializing the sum_action_step matrix
        # 1st index: action (1 to 6)
        # 2nd index: time step (1 to FIXED_NB_STEPS)
        # Value inside: number of time this action has been chosen, divided by number of runs
    sum_action_step = []
    for i in range(6):
        a = ([0.0 for k in range(FIXED_NB_STEPS)])
        sum_action_step.append(a)

    # Running the MainLoop FIXED_NB_STEPS time and populating the sum_action_step matrix
    for i in range(NB_RUNS):
        main_loop = MainLoop()
        result = main_loop.findBestArm()
        action_mem = main_loop.get_action_memory()
        for time_step in range(len(action_mem)):
            action = action_mem[time_step]
            sum_action_step[action-1][time_step] += 1.0/NB_RUNS
        # Show at which step we are
        if i % 10 == 0:
            print(i)
    #print(str(sum_action_step)) 

    # Drawing the results
    drawer = Drawer(EXP_NAME)
    drawer.save_png(range(FIXED_NB_STEPS), sum_action_step, "Number of pulls", "P(I_t = i)", "ActionEliminationSampling")
    drawer.save_csv(sum_action_step, "ActionEliminationSampling")

