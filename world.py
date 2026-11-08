"""Metric planar navigation benchmark. Controls are target speed and yaw rate."""
import math
import torch

DT = 0.25
RADIUS = 0.18


def motor(velocity, command, terrain):
    v, w = velocity.unbind(-1)
    drive, turn = command.unbind(-1)
    grip = terrain
    dv = (1.05 + 2.8*grip) * (drive*(.70+.30*grip) - v) - .10*v*turn.abs()
    dw = (1.25 + 2.1*grip) * (turn*(.43+.57*grip) - w) - .16*v*w
    return torch.stack((v + DT*dv, w + DT*dw), -1)


def advance(pose, vel):
    x, y, theta = pose.unbind(-1)
    speed, yaw = vel.unbind(-1)
    mid = theta + DT*yaw/2
    return torch.stack((x+DT*speed*mid.cos(), y+DT*speed*mid.sin(), theta+DT*yaw), -1)


def clearance(start, end, circles):
    """Swept disk against closed circles; negative means contact."""
    d = (end-start).unsqueeze(-2)
    offset = circles[..., :2] - start.unsqueeze(-2)
    t = ((d*offset).sum(-1) / d.square().sum(-1).clamp_min(1e-9)).clamp(0,1)
    return ((offset-t.unsqueeze(-1)*d).norm(dim=-1)-circles[..., 2]-RADIUS).amin(-1)


def scenarios(n, seed, device):
    gen = torch.Generator().manual_seed(seed)
    jitter = torch.rand(n, 7, generator=gen)*2-1
    sign = torch.where(jitter[:, 0]>0, 1., -1.)
    pose = torch.stack((jitter[:,1]*.1, jitter[:,2]*.22, jitter[:,3]*.16), -1)
    gates = torch.stack((torch.stack((2.9+jitter[:,1]*.15, sign*(.95+jitter[:,4]*.24)), -1),
                         torch.stack((5.9+jitter[:,2]*.15, -sign*(.70+jitter[:,5]*.24)), -1)), 1)
    circles = torch.stack((torch.stack((1.50+jitter[:,3]*.22, sign*(.36+jitter[:,5]*.16), .43+jitter[:,6]*.09), -1),
                           torch.stack((4.38+jitter[:,6]*.2, sign*(.13+jitter[:,4]*.18), .40+jitter[:,1]*.08), -1)), 1)
    return pose.to(device), gates.to(device), circles.to(device)
