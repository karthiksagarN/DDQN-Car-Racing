import GameEnv
import pygame
import numpy as np
from ddqn_keras import DDQNAgent
from collections import deque
import random, math
TOTAL_GAMETIME = 1000 # Max game time for one episode
N_EPISODES = 3000
REPLACE_TARGET = 50 
game = GameEnv.RacingEnv()
game.fps = 60
GameTime = 0 
GameHistory = []
renderFlag = False
ddqn_agent = DDQNAgent(alpha= 0.0005, gamma=0.99, n_actions=5, epsilon=1.00, epsilon_end=0.10, epsilon_dec=0.9995, replace_target= REPLACE_TARGET, batch_size=512, input_dims=19)
# if you want to load the existing model uncomment this line.
# careful an existing model might be overwritten
#ddqn_agent.load_model()
ddqn_scores = []
eps_history = []
import numpy as np
import pygame
import pickle

def run():
    ddqn_scores = []
    eps_history = []

    for e in range(N_EPISODES):
        game.reset()  # reset environment
        done = False
        score = 0
        counter = 0
        observation_, reward, done = game.step(0)
        observation = np.array(observation_)
        gtime = 0  # game time
        renderFlag = False

        if e % 10 == 0 and e > 0:
            renderFlag = True

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            action = ddqn_agent.choose_action(observation)
            observation_, reward, done = game.step(action)
            observation_ = np.array(observation_)

            # Timeout if no reward for 100 frames
            if reward == 0:
                counter += 1
                if counter > 100:
                    done = True
            else:
                counter = 0

            score += reward
            ddqn_agent.remember(observation, action, reward, observation_, int(done))
            observation = observation_
            ddqn_agent.learn()

            gtime += 1
            if gtime >= TOTAL_GAMETIME:
                done = True

            if renderFlag:
                game.render(action)

        # Log metrics
        eps_history.append(ddqn_agent.epsilon)
        ddqn_scores.append(score)
        avg_score = np.mean(ddqn_scores[max(0, e-100):(e+1)])

        if e % REPLACE_TARGET == 0 and e > REPLACE_TARGET:
            ddqn_agent.update_network_parameters()

        if e % 10 == 0 and e > 10:
            ddqn_agent.save_model()
            print("Model saved")

        print(f"Episode: {e} | Score: {score:.2f} | Avg Score: {avg_score:.2f} | "
              f"Epsilon: {ddqn_agent.epsilon:.4f} | Memory: {ddqn_agent.memory.mem_cntr % ddqn_agent.memory.mem_size}")

    # Save logs for plotting
    with open("logs.pkl", "wb") as f:
        pickle.dump((ddqn_scores, eps_history), f)
  
run()        
