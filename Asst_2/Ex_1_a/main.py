import gym
import time

from algorithms import *
from utils import *
import matplotlib.pyplot as plt
import csv
import os
import copy



NB_RUNS = 10
NB_SEGMENTS = 100
NB_EPISODES = 10
ACTIONLIST = [0, 1, 2, 3, 4, 5]
TEMPERATURE = 0.01
GAMMA = 1
ALPHA = 0.5
PARAMS = {"action_list": ACTIONLIST, "temperature": TEMPERATURE, "alpha":ALPHA, "gamma":GAMMA}
ENVIRONMENT = gym.make('Taxi-v2')
ALGORITHM = Sarsa(ENVIRONMENT, PARAMS)


class Drawer():
    def __init__(self, exp_name):
        self.output_path_root = "./experiments/" + exp_name
        self.makeDir("./experiments")
        self.makeDir(self.output_path_root)

    def savePickProbPNG(self, x, sum_action_step, x_label, y_label, plot_title):
        plt.subplots()
        nb_arms = len(sum_action_step)
        for i in range(nb_arms):
            y = sum_action_step[i]
            y = smoothen(y)
            plt.plot(x, y)
        plt.title(plot_title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        output_path = self.output_path_root + "/" + plot_title + ".png"
        plt.savefig(output_path, bbox_inches="tight")

    def savePlotPNG(self, x, y, x_label, y_label, plot_title):
        plt.subplots()
        plt.plot(x,y)
        plt.title(plot_title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        output_path = self.output_path_root + "/" + plot_title + ".png"
        plt.savefig(output_path, bbox_inches="tight")

    def saveMultiPlotPNG(self, x, y_list, x_label, y_label, plot_title, legend = False):
        plt.subplots()
        for y_id in range(len(y_list)):
            y = y_list[y_id]
            y_smooth = smoothen(y)
            if legend:
                plt.plot(x,y, label = legend[y_id])
        if legend:
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True)
        plt.title(plot_title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        output_path = self.output_path_root + "/" + plot_title + ".png"
        plt.savefig(output_path, bbox_inches="tight")

    def saveCSV(self, sum_action_step, csv_title):
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
    
    def saveMultiCSV(self, csv_title, list_of_lists, legend):
        path = self.output_path_root + "/" + csv_title + ".csv"
        assert len(legend) == len(list_of_lists), "Error writing CSV " + csv_title + ": legend has to have same size than list_of_lists"
        with open(path, "w") as csvfile:
            writer = csv.writer(csvfile)
            i = 0
            for a_list in list_of_lists:
                b_list = [legend[i]] + a_list
                writer.writerow(b_list)
                i += 1

    def makeDir(self, path):
        # Check if directory exists, if not, create it
        if not os.path.exists(path):
            try:  
                os.mkdir(path)
            except OSError:  
                print ("Creation of the directory %s failed" % path)
            else:  
                print ("Successfully created the directory %s " % path) 


class Trainer(object):
    def __init__(self, env, algo, drawer):
        self.env = env
        self.algo = algo
        self.drawer = drawer


    def trainOneEpisode(self):
        self.algo.partialReset()
        done = 0
        state = self.env.reset()
        action = self.algo.nextAct(state)
        while not done:
            next_state, reward, done, _ = self.env.step(action)
            next_action = self.algo.nextAct(next_state)
            self.algo.update([state, action, reward, next_state, next_action])
            state = next_state
            action = next_action
        return self.algo.getReturn()

    def testOneEpisode(self):
        self.algo.partialReset()
        done = 0
        state = self.env.reset()
        action = self.algo.nextAct(state)
        while not done:
            next_state, reward, done, _ = self.env.step(action)
            next_action = self.algo.nextGreedyAct(next_state)
            self.algo.updateNoLearn(reward)
            #self.algo.update([state, action, reward, next_state, next_action])
            state = next_state
            action = next_action
        return self.algo.getReturn()

    def evalAvgReturn(self):
        avg_train_ret = 0
        avg_test_ret = 0
        for run in range(NB_RUNS):
            self.algo.reset()
            for seg in range(NB_SEGMENTS - 1):
                print("Training run:" + str(run) + ", segment: " + str(seg))
                # Training episodes
                for episode in range(NB_EPISODES):
                    self.trainOneEpisode()

            # Getting "training data" in the last segment
            print("Getting training performance at run:" + str(run))
            for episode in range(NB_EPISODES):
                train_ret = self.trainOneEpisode()
                if train_ret < 0:
                    print("*** NEGATIVE TRAINING PERFORMANCE: " + str(train_ret))
                avg_train_ret += (train_ret - avg_train_ret)/(run*NB_EPISODES + episode + 1)
            print("Training performance average:" + str(avg_train_ret))


            # Geting "testing data" with greedy policy after training
            print("Getting testing performance at run:" + str(run))
            for episode in range(10):
                test_ret = self.testOneEpisode()
                if test_ret < 0:
                    print("*** NEGATIVE TESTING PERFORMANCE: " + str(test_ret))
                avg_test_ret += (test_ret - avg_test_ret)/(run*10 + episode + 1)        
            print("Testing performance average:" + str(avg_test_ret))

if __name__ == '__main__':
    exp_name = "Try_2"
    env = ENVIRONMENT
    drawer = Drawer(exp_name)
    avg_train_ret = 0
    avg_test_ret = 0
    algo = ALGORITHM

    trainer = Trainer(env, algo, drawer)

    trainer.evalAvgReturn()



   # drawer.savePlotPNG(range(NB_SEGMENTS),avg_ret, "Episode", "Average return", "Average return on Taxi using algo: " + algo.getName() + ", temp: " + str(TEMPERATURE) + ", learning rate: " + str(ALPHA))
 
