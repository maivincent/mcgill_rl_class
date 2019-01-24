from environment import Environment
from algorithms import ActionEliminationAlgo


FIXED_NB_STEPS = 6000

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




if __name__ == "__main__":
    action_sum_per_step = [[0, 0, 0, 0, 0, 0] for k in range(FIXED_NB_STEPS)]

    for i in range(1000):
        main_loop = MainLoop()
        result = main_loop.findBestArm()
        action_mem = main_loop.get_action_memory()
        for time_step in range(len(action_mem)):
            action = action_mem[time_step]
            try:
                action_sum_per_step[time_step-1][action-1] += 1
            except:
                print("Could not with time step: " + str(time_step) + " and action: " + str(action))
        print(result)
    print action_sum_per_step

