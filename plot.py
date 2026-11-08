"""Plot paired trajectories and online grip inference directly from the saved logs."""
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from dynamics import OUT


def main():
    traces = json.loads((OUT/"traces.json").read_text())
    report = json.loads((OUT/"navigation.json").read_text())
    colors = {"reactive":"#a83c3a","fixed":"#3877b5","adaptive":"#16836e"}
    fig, (ax, belief) = plt.subplots(2,1,figsize=(8,6.5),constrained_layout=True)
    for method, color in colors.items():
        key = "switch/"+method
        tr = traces[key]
        steps = report["trials"][key][0]["steps"]
        xy = np.asarray(tr["positions"])[:steps+1,0]
        ax.plot(*xy.T,color=color,label=method,linewidth=2)
        ax.scatter(*xy[-1],s=24,color=color)
    for x,y,radius in tr["circles"][0]:
        ax.add_patch(Circle((x,y),radius,color="#46505b"))
        ax.add_patch(Circle((x,y),radius+.18,color="#46505b",alpha=.13))
    for i,(x,y) in enumerate(tr["goals"][0],1):
        ax.add_patch(Circle((x,y),.32,fill=False,ls="--"))
        ax.text(x,y,str(i),ha="center",va="center")
    ax.axvline(2.8,ls=":",color="#888")
    ax.set(aspect="equal",xlabel="x (m)",ylabel="y (m)",title="Ordered waypoint navigation · held-out route 0")
    ax.legend(loc="best",ncol=3)
    series=np.asarray(tr["grip"])[:steps,0]
    belief.step(np.arange(steps)*.25,series[:,0],where="post",label="simulator grip",color="#343c44")
    belief.plot(np.arange(steps)*.25,series[:,1],label="online estimate before action",color=colors["adaptive"])
    belief.set(ylim=(0,1.05),xlabel="Simulation time (s)",ylabel="Grip parameter",title="Tracking a surface change from noisy wheel-speed feedback")
    belief.legend()
    fig.savefig(OUT/"navigation.png",dpi=170)


if __name__=="__main__":
    main()
