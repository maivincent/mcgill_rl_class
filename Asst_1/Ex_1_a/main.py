from environment import Environment
from algorithms import ActionEliminationAlgo



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
        while not self.algo.isDone():
            self.doOneStep()
        return self.algo.result()




if __name__ == "__main__":
    main_loop = MainLoop()
    result = main_loop.findBestArm()

