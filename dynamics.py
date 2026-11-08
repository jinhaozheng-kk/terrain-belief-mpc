"""Supervised identification from complete, independently seeded rollouts."""
import json
from pathlib import Path

import torch
from torch import nn
from world import advance, motor

OUT = Path("results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Dynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(5,48), nn.Tanh(), nn.Linear(48,48), nn.Tanh(), nn.Linear(48,2))

    def forward(self, vel, command, grip):
        x = torch.cat((vel, command, grip.unsqueeze(-1)), -1)
        return vel + self.network(x)


def collect(count, seed):
    rng = torch.Generator().manual_seed(seed)
    grip = .15+.85*torch.rand(count, generator=rng)
    v = torch.zeros(count,2)
    states, actions = [v], []
    for t in range(32):
        if t%3 == 0:
            u = torch.rand(count,2, generator=rng)*torch.tensor([1.35,4.4])-torch.tensor([0.,2.2])
        v = motor(v,u,grip)
        actions.append(u)
        states.append(v)
    return tuple(t.to(DEVICE) for t in (torch.stack(states),torch.stack(actions),grip))


def main():
    OUT.mkdir(exist_ok=True)
    torch.manual_seed(91)
    x, u, g = collect(512, 91)
    model = Dynamics().to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=.002)
    for _ in range(900):
        t = torch.randint(32,(384,),device=DEVICE)
        i = torch.randint(512,(384,),device=DEVICE)
        pred = model(x[t,i],u[t,i],g[i])
        loss = (pred-x[t+1,i]).square().mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
    torch.save(model.state_dict(), OUT/"dynamics.pt")
    x,u,g = collect(128, 3029)
    with torch.no_grad():
        one = model(x[:-1],u,g.expand(32,-1))
        v = x[0]
        truth = torch.zeros(len(g),3,device=DEVICE)
        pred_pose = truth.clone()
        for k in range(12):
            v = model(v,u[k],g)
            pred_pose = advance(pred_pose,v)
            truth = advance(truth,x[k+1])
        report = dict(train_transitions=512*32, test_trajectories=128,
                      one_step_rmse=(one-x[1:]).square().mean((0,1)).sqrt().tolist(),
                      horizon12_mean_position_error_m=(truth[:,:2]-pred_pose[:,:2]).norm(dim=-1).mean().item(),
                      device=DEVICE, torch=str(torch.__version__))
    (OUT/"identification.json").write_text(json.dumps(report,indent=2))
    print(report)


if __name__ == "__main__":
    main()
