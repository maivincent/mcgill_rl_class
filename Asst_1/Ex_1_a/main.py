from environment import Environment
from algorithms import ActionEliminationAlgo
import matplotlib.pyplot as plt
import os


FIXED_NB_STEPS = 6000
NB_RUNS = 5000
EXP_NAME = "action_elimination_article"

class MainLoop():
    def __init__(self):
        self.delta = 0.1
        self.epsilon = 0.01
        self.algo = ActionEliminationAlgo(self.delta, self.epsilon)
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
        #while not self.algo.isDone():
        while self.step < FIXED_NB_STEPS:
            self.doOneStep()
            #print("Step: " + str(self.step))
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

    def make_dir(self, path):
        if not os.path.exists(path):
            try:  
                os.mkdir(path)
            except OSError:  
                print ("Creation of the directory %s failed" % path)
            else:  
                print ("Successfully created the directory %s " % path)        






if __name__ == "__main__":
    sum_action_step = []
    for i in range(6):
        a = ([0 for k in range(FIXED_NB_STEPS)])
        sum_action_step.append(a)


    for i in range(NB_RUNS):
        main_loop = MainLoop()
        result = main_loop.findBestArm()
        action_mem = main_loop.get_action_memory()
        for time_step in range(len(action_mem)):
            action = action_mem[time_step]
            try:
                sum_action_step[action-1][time_step] += 1/NB_RUNS
            except:
                print("Could not with time step: " + str(time_step) + " and action: " + str(action))
        if i % 10 == 0:
            print(i)
    print sum_action_step

    # Drawing the results
    drawer = Drawer(EXP_NAME)
    drawer.save_png(range(FIXED_NB_STEPS), sum_action_step, "Number of pulls", "P(I_t = i)", "ActionEliminationSampling")

