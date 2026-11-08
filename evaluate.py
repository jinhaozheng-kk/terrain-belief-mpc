"""Paired navigation trials. The hidden grip enters only the simulated plant and oracle."""
import argparse
import json
from pathlib import Path

import torch

from dynamics import DEVICE, Dynamics, OUT
from planner import TerrainFilter, plan
from world import advance, clearance, motor, scenarios


@torch.inference_mode()
def run(model, method, condition, count, seed):
    pose, goals, obstacles = scenarios(count,seed,DEVICE)
    rng = torch.Generator(device=DEVICE).manual_seed(seed+1)
    sensor = torch.Generator(device=DEVICE).manual_seed(seed+2)
    velocity = torch.zeros_like(pose[:,:2])
    observed = velocity.clone()
    filt = TerrainFilter(count,DEVICE)
    stage = torch.zeros(count,device=DEVICE,dtype=torch.long)
    hit = torch.zeros_like(stage,dtype=torch.bool)
    ticks = torch.zeros_like(stage)
    distance = torch.zeros_like(velocity[:,0])
    ids = torch.arange(count,device=DEVICE)
    warm = None
    histories = [pose[:,:2].cpu().tolist()]
    beliefs = []
    for step in range(64):
        active = ~hit & (stage<2)
        if not active.any():
            break
        target = goals[ids,stage.clamp_max(1)]
        actual = torch.full_like(distance,.90 if condition=="dry" else .21)
        if condition == "switch":
            actual = torch.where(pose[:,0]<2.8, .90, .21)
        grip = filt.mean() if method=="adaptive" else torch.full_like(actual,.9)
        if method=="oracle":
            grip = actual
        if method=="reactive":
            delta = target-pose[:,:2]
            angle = torch.atan2(delta[:,1],delta[:,0])-pose[:,2]
            yaw = torch.atan2(angle.sin(),angle.cos())
            u = torch.stack((1.15*yaw.cos().clamp(0,1), (2*yaw).clamp(-2.2,2.2)),dim=1)
        else:
            u,warm = plan(model,pose,observed,target,obstacles,grip,warm,rng)
        vel = motor(velocity,u,actual)
        nxt = advance(pose,vel)
        noisy = vel + torch.randn(vel.shape,device=DEVICE,generator=sensor)*vel.new_tensor([.012,.018])
        if method=="adaptive":
            filt.update(model,observed,u,noisy)
        beliefs.append(torch.stack((actual,grip),-1).cpu().tolist())
        gap = clearance(pose[:,:2],nxt[:,:2],obstacles)
        distance += (nxt[:,:2]-pose[:,:2]).norm(dim=-1)*active
        hit |= (gap<=0)&active
        stage += ((nxt[:,:2]-target).norm(dim=-1)<.32)&active&~hit
        ticks += active
        pose = torch.where(active[:,None],nxt,pose)
        velocity,observed = vel,noisy
        histories.append(pose[:,:2].cpu().tolist())
    rows = [dict(route=i, reached=int(stage[i]), collision=bool(hit[i]), steps=int(ticks[i]),
                 distance_m=float(distance[i])) for i in range(count)]
    trace = dict(goals=goals.cpu().tolist(), circles=obstacles.cpu().tolist(), positions=histories, grip=beliefs)
    return rows,trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes",type=int,default=32)
    parser.add_argument("--seed",type=int,default=7193)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model = Dynamics().to(DEVICE)
    model.load_state_dict(torch.load(OUT/"dynamics.pt",map_location=DEVICE,weights_only=True))
    model.eval()
    summary, trials, traces = {}, {}, {}
    for condition in ("dry","slick","switch"):
        for method in ("reactive","fixed","adaptive","oracle"):
            key = f"{condition}/{method}"
            rows,trace = run(model,method,condition,args.routes,args.seed)
            trials[key] = rows
            summary[key] = dict(success=sum(r["reached"]==2 for r in rows), collision=sum(r["collision"] for r in rows),
                                timeout=sum(r["reached"]<2 and not r["collision"] for r in rows))
            # Retain full traces for six routes, preserving route index pairing.
            traces[key] = {k: ([step[:6] for step in v] if k in ("positions","grip") else v[:6]) for k,v in trace.items()}
            print(key,summary[key],flush=True)
    (OUT/"navigation.json").write_text(json.dumps(dict(routes=args.routes,seed=args.seed,summary=summary,trials=trials),indent=2))
    (OUT/"traces.json").write_text(json.dumps(traces,separators=(",",":")))


if __name__=="__main__":
    main()
