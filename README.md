
<p align="center">
    <img src="https://raw.githubusercontent.com/aidudezzz/deepbots-swag/main/logo/deepworlds_full.png">
</p>

<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-6-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

Deepworlds is a support repository for the [deepbots framework](https://github.com/aidudezzz/deepbots),
containing examples of the framework's usage on the [Webots](https://www.cyberbotics.com) robot simulator.

If the following sections feel overwhelming, feel free to start on our [deepbots-tutorials repository](https://github.com/aidudezzz/deepbots-tutorials)
for a beginner's in-depth introduction to the way the deepbots framework is used.

## Run an example in Webots

1. Clone the repository using:
   ```
   git clone https://github.com/aidudezzz/deepworlds.git
   ```

2. Install specific packages for each example you want to use by running the following:
   ```
   pip install -r <path to requirements file>
   ``` 
   You can find the requirement files on the `/requirements/<example-name>.txt` path of each example, 
   e.g., [/examples/cartpole/cartpole_discrete/requirements/](./examples/cartpole/cartpole_discrete/requirements).

3. Through Webots, open the .wbt file of the example you are interested in and hit run to train the provided agent. 
You can find the .wbt files under `/worlds/`, e.g., [/examples/cartpole/cartpole_discrete/worlds/](./examples/cartpole/cartpole_discrete/worlds).

For more information on the examples, refer to each one's README, and examine the code within their `/controllers/` directory. 

## Some important notes

Each example might be split into discrete and continuous action space cases. The reason for this split is that depending on the action space, 
different kinds of reinforcement learning agents need to be used, and thus quite large changes are needed in the code. 

Keep in mind that each example can have multiple solutions provided using the two schemes of deepbots 
([robot supervisor](https://github.com/aidudezzz/deepbots#combined-robot-supervisor-scheme) and 
[emitter-receiver](https://github.com/aidudezzz/deepbots#emitter---receiver-scheme)) and with different reinforcement learning agents, backends, etc.

We suggest starting your exploration from the **discrete cartpole example using the robot supervisor scheme**, 
as it is also the example used in the tutorial. The main class/controller implementation can be found 
[here](./examples/cartpole/cartpole_discrete/controllers/robot_supervisor_manager/robot_supervisor.py), and the corresponding 
tutorial to create it from scratch is [here](https://github.com/aidudezzz/deepbots-tutorials/blob/master/robotSupervisorSchemeTutorial/README.md).

## Directories structure

```
\deepworlds
    \examples
        \cartpole
            \cartpole_discrete
                \controllers
                \requirements
                \worlds
            \cartpole_continuous
                \controllers
                \requirements
                \worlds
            \...
        \find_and_avoid
            \find_and_avoid_continuous
                \controllers
                \requirements
                \worlds
            \...
        \pit_escape
            \pit_escape_discrete
                \controllers
                \requirements
                \worlds
            \...
        \(more examples)       
```

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/tsampazk"><img src="https://avatars.githubusercontent.com/u/27914645?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Kostas Tsampazis</b></sub></a><br /><a href="https://github.com/aidudezzz/deepworlds/issues?q=author%3Atsampazk" title="Bug reports">🐛</a> <a href="https://github.com/aidudezzz/deepworlds/commits?author=tsampazk" title="Code">💻</a> <a href="https://github.com/aidudezzz/deepworlds/commits?author=tsampazk" title="Documentation">📖</a> <a href="#example-tsampazk" title="Examples">💡</a> <a href="#ideas-tsampazk" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-tsampazk" title="Maintenance">🚧</a> <a href="#projectManagement-tsampazk" title="Project Management">📆</a> <a href="#question-tsampazk" title="Answering Questions">💬</a> <a href="https://github.com/aidudezzz/deepworlds/pulls?q=is%3Apr+reviewed-by%3Atsampazk" title="Reviewed Pull Requests">👀</a></td>
    <td align="center"><a href="http://eakirtas.webpages.auth.gr/"><img src="https://avatars.githubusercontent.com/u/10010230?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Manos Kirtas</b></sub></a><br /><a href="https://github.com/aidudezzz/deepworlds/issues?q=author%3AManosMagnus" title="Bug reports">🐛</a> <a href="https://github.com/aidudezzz/deepworlds/commits?author=ManosMagnus" title="Code">💻</a> <a href="https://github.com/aidudezzz/deepworlds/commits?author=ManosMagnus" title="Documentation">📖</a> <a href="#example-ManosMagnus" title="Examples">💡</a> <a href="#ideas-ManosMagnus" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-ManosMagnus" title="Maintenance">🚧</a> <a href="#projectManagement-ManosMagnus" title="Project Management">📆</a> <a href="#question-ManosMagnus" title="Answering Questions">💬</a> <a href="https://github.com/aidudezzz/deepworlds/pulls?q=is%3Apr+reviewed-by%3AManosMagnus" title="Reviewed Pull Requests">👀</a></td>
    <td align="center"><a href="https://github.com/rohit-kumar-j"><img src="https://avatars.githubusercontent.com/u/37873142?v=4?s=100" width="100px;" alt=""/><br /><sub><b>RKJ</b></sub></a><br /><a href="#ideas-rohit-kumar-j" title="Ideas, Planning, & Feedback">🤔</a></td>
    <td align="center"><a href="https://github.com/wakeupppp"><img src="https://avatars.githubusercontent.com/u/39750646?v=4?s=100" width="100px;" alt=""/><br /><sub><b>wakeupppp</b></sub></a><br /><a href="https://github.com/aidudezzz/deepworlds/issues?q=author%3Awakeupppp" title="Bug reports">🐛</a></td>
    <td align="center"><a href="https://www.linkedin.com/in/kelvin-yang-b7b508198/"><img src="https://avatars.githubusercontent.com/u/49781698?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Jiun Kai Yang</b></sub></a><br /><a href="https://github.com/aidudezzz/deepworlds/commits?author=KelvinYang0320" title="Code">💻</a> <a href="https://github.com/aidudezzz/deepworlds/commits?author=KelvinYang0320" title="Documentation">📖</a> <a href="#example-KelvinYang0320" title="Examples">💡</a> <a href="#ideas-KelvinYang0320" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/aidudezzz/deepworlds/pulls?q=is%3Apr+reviewed-by%3AKelvinYang0320" title="Reviewed Pull Requests">👀</a> <a href="#maintenance-KelvinYang0320" title="Maintenance">🚧</a> <a href="#projectManagement-KelvinYang0320" title="Project Management">📆</a> <a href="https://github.com/aidudezzz/deepworlds/issues?q=author%3AKelvinYang0320" title="Bug reports">🐛</a> <a href="#question-KelvinYang0320" title="Answering Questions">💬</a></td>
    <td align="center"><a href="https://nickkok.github.io/my-website/"><img src="https://avatars.githubusercontent.com/u/8222731?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Nikolaos Kokkinis-Ntrenis</b></sub></a><br /><a href="https://github.com/aidudezzz/deepworlds/commits?author=NickKok" title="Code">💻</a> <a href="https://github.com/aidudezzz/deepworlds/commits?author=NickKok" title="Documentation">📖</a> <a href="#example-NickKok" title="Examples">💡</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

<b> Special thanks to <a href='https://www.papanikolaouev.com/'>Papanikolaou Evangelia</a> </b> for designing project's logo! </b> 
