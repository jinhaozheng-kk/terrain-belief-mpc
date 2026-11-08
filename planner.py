"""Receding-horizon CEM planner and a finite-state online terrain filter."""
import torch
from world import advance, clearance

HORIZON = 12
SAMPLES = 96
ELITES = 12


class TerrainFilter:
    def __init__(self, batch, device):
        self.grid = torch.linspace(.15, 1., 18, device=device)
        self.prob = torch.full((batch, len(self.grid)), 1/len(self.grid), device=device)

    def mean(self):
        return (self.grid*self.prob).sum(1)

    def update(self, model, velocity, command, observed):
        shape = (*self.prob.shape, 2)
        expected = model(velocity[:,None].expand(shape), command[:,None].expand(shape), self.grid.expand_as(self.prob))
        scale = observed.new_tensor([.035, .06])
        err = ((expected-observed[:,None])/scale).square().sum(-1)
        # Small reset probability permits changes after crossing a surface boundary.
        prior = .95*self.prob + .05/len(self.grid)
        self.prob = (prior.log()-.5*err).softmax(1)


def plan(model, pose, velocity, target, circles, grip, warm, generator):
    batch = len(pose)
    reference = pose.new_tensor([1.05, 0.])
    mean = reference.expand(batch,4,2) if warm is None else warm[:,::3]
    std = pose.new_tensor([.5,1.3]).expand_as(mean)
    for _ in range(2):
        noise = torch.randn(batch,SAMPLES,4,2,device=pose.device,generator=generator)
        knots = mean[:,None] + std[:,None]*noise
        knots[...,0].clamp_(0,1.35)
        knots[...,1].clamp_(-2.2,2.2)
        knots[:,0] = mean
        pos = pose[:,None].expand(-1,SAMPLES,-1)
        vel = velocity[:,None].expand_as(pos[...,:2])
        score = torch.zeros_like(pos[...,0])
        for t in range(HORIZON):
            action = knots[:,:,t//3]
            vel = model(vel, action, grip[:,None].expand_as(score))
            nxt = advance(pos, vel)
            gap = clearance(pos[...,:2], nxt[...,:2], circles[:,None])
            dist = (nxt[...,:2]-target[:,None]).norm(dim=-1)
            score += .35*dist + 12*(.12-gap).clamp_min(0) + 28*(gap<=0) + .015*action[...,1].square()
            pos = nxt
        score += 4*dist
        indices = score.topk(ELITES, largest=False).indices
        elite = knots.gather(1,indices[:,:,None,None].expand(-1,-1,4,2))
        mean = elite.mean(1)
        std = elite.std(1).clamp_min(.12)
    # Execute the best sampled feasible tradeoff; averaging paths can cut through obstacles.
    best = knots[torch.arange(batch,device=pose.device), indices[:,0]].repeat_interleave(3,dim=1)
    return best[:,0], torch.cat((best[:,1:],best[:,-1:]),dim=1)
