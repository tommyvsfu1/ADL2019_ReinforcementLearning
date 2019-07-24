# Report  
## Policy Gradient  
### Baseline
- [x] Getting averaging reward in 30 episodes over 0 in LunarLander

###   structure 1
```python
nn.Sequential(
    nn.Linear(state_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, action_num),
    nn.Softmax(dim=-1)
)
```
*   Testing  (pg_best.cpt)  
Run 30 episodes
Mean: 196.97405585001098

*  training curve
![](https://i.imgur.com/TBSnXqJ.png) 


