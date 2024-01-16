# 2024 MLDS Long Competition

This is the repository for the 2024 MLDS long competition!

Here is the presentation introducing it: https://docs.google.com/presentation/d/1yyHxMkUn4X3pWK06-q3J2rZ2-Q5HZSMq/edit?usp=sharing&ouid=106156873976907460505&rtpof=true&sd=true

And here is a quick start presentation: https://docs.google.com/presentation/d/1tGScW-0R_RbkbZYtsg-6iVtEtAAfPg_p/edit?usp=sharing&ouid=106156873976907460505&rtpof=true&sd=true

And here is the Colab version: https://colab.research.google.com/drive/1l9k4F9vezXvMWutXu7rKy00mwv75NZxg?usp=sharing

After cloning this repository, start by making sure that you have all the dependencies required. You can do so by running `pip install -r requirements.txt`. Also note that you will need Python 3.9 or above. We recommend doing all of this in a virtual environment.

Now you should be able to run a sample battle by running "main.py." For creating your own battle algorithm, you can start out by modifying the myAgent() function in "main.py" (or rather, by creating a new function to act as a new agent). However, for more complex algorithms such as those using reinforcement learning, you should be able to create a new class, subclassing the Agent class from "agent.py" in your own file.

By default, the game opens up a window and runs at a defined frame rate. If you want to train agents as fast as possible without this limitation, make sure to pass `None` for the `render_mode` argument of the method `run_game`; i.e.:

```py
env.run_game(200, render_mode=None)
```