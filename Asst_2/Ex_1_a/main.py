import gym
import time

from algorithms import *
from utils import *
import matplotlib.pyplot as plt
import csv
import os
import copy



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

    def testOneEpisode(self, learn = False, render = False):
        self.algo.partialReset()
        done = 0
        state = self.env.reset()
        action = self.algo.nextGreedyAct(state)
        if render: 
            print(" -------- New episode -------")
            env.render()
        while not done:
            next_state, reward, done, _ = self.env.step(action)
            if render: 
                env.render()
            next_action = self.algo.nextGreedyAct(next_state)
            if learn: self.algo.update([state, action, reward, next_state, next_action])
            else: self.algo.updateNoLearn(reward)
            state = next_state
            action = next_action
        ret = self.algo.getReturn()
        return self.algo.getReturn()

    def evalAvgReturn(self, numbers, learn):
        nb_runs = numbers[0]
        nb_segments = numbers[1]
        nb_episodes = numbers[2]
        avg_train_ret = 0
        avg_test_ret = 0
        for run in range(nb_runs):
            print("Training run:" + str(run))

            self.algo.reset()
            for seg in range(nb_segments - 1):
                # Training episodes
                for episode in range(nb_episodes):
                    self.trainOneEpisode()
                self.testOneEpisode()

            # Getting "training data" in the last segment
            for episode in range(nb_episodes):
                train_ret = self.trainOneEpisode()
                avg_train_ret += (train_ret - avg_train_ret)/(run*nb_episodes + episode + 1)


            # Geting "testing data" with greedy policy after training
            for episode in range(1):
                test_ret = self.testOneEpisode(learn = learn)
                avg_test_ret += (test_ret - avg_test_ret)/(run*10 + episode + 1)        

        return avg_train_ret, avg_test_ret

if __name__ == '__main__':
    exp_name = "Try_2"
    env = gym.make('Taxi-v2')
    action_list = [0, 1, 2, 3, 4, 5]


    nb_runs = 10
    nb_segments = 100
    nb_episodes = 10
    gamma = 1
    
    alpha = 0.5
    temperature = 1
    learn = True


    for temp
    algo_params = {"action_list": action_list, "temperature": temperature, "alpha":alpha, "gamma":gamma}
    algo = Sarsa(env, algo_params)

    drawer = Drawer(exp_name)
   
    trainer = Trainer(env, algo, drawer)
    trainer.evalAvgReturn([nb_runs, nb_segments, nb_episodes], learn)





   # drawer.savePlotPNG(range(NB_SEGMENTS),avg_ret, "Episode", "Average return", "Average return on Taxi using algo: " + algo.getName() + ", temp: " + str(TEMPERATURE) + ", learning rate: " + str(ALPHA))
 
