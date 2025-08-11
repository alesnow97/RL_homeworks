import torch
import gym
import gym.spaces

print("Testing PyTorch and Gym installation...")
try:
    # Check if PyTorch is installed correctly
    x = torch.rand(5, 3)
    print("PyTorch is working. Random tensor:", x)

    # Check if Gym is installed correctly
    env = gym.make('CartPole-v1')
    
    observation = env.reset()
    print("Gym is working. Initial observation:", observation)

except Exception as e:
    print("Error occurred:", e)

print(torch.__version__)