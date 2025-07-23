# Todo:
1- Import Video from logger:
in training script:
```python
from stable_baselines3.common.logger import HParam, Video
```
2- Add necessary variables to AdditionalInfoCallback class:
add in the constructor:
```python
        self.frames = []  # List to store frames for GIF creation
        self.episode_cnt = 1
        self.record = False
        self.render_interval=render_interval
```
3- Implement the on_step() event handler to record frames periodically:
```python
if self.env.done:
            if self.episode_cnt % self.render_interval == 0:
                self.env.camera.enable(self.env.timestep * 10) # basic time step = 32
                self.record = True
            else:
                self.env.camera.disable()
                self.record = False
            self.episode_cnt+=1
            self.frames = []
            print(f'Starting Episode {self.episode_cnt} ...')  
```
4- Add video creation and logging to tensorboard to on_rollout_end() event:
if self.record:
```python
            # Save the frames to tensorboard
            frame = self.env.render(mode='rgb_array') # (c, h, w)
            self.frames.append(frame) 
            video = np.asarray([self.frames])
            self.logger.record("visualization",
                                Video(torch.from_numpy(video), fps=30),
                                exclude=("stdout", "log", "json", "csv"))
```
5- Add render parameters to run() function:
```python
run(... ,
        log_interval=4,
        render_interval=100)
```
# Results:
"Visualization" tab is added to Tensorboard monitoring & is updated after a predefined sequence of episodes, hence more ability to track the agent performance across different difficulty levels.