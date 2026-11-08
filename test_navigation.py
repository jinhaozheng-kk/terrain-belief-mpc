import unittest
import torch
from planner import TerrainFilter
from world import advance, clearance, motor, scenarios


class GeometryTests(unittest.TestCase):
    def test_swept_contact(self):
        circles=torch.tensor([[.5,0.,.08]])
        self.assertLess(clearance(torch.tensor([0.,0.]),torch.tensor([1.,0.]),circles).item(),0)

    def test_stationary_clearance(self):
        point=torch.tensor([0.,0.])
        self.assertAlmostEqual(clearance(point,point,torch.tensor([[1.,0.,.2]])).item(),.62,places=5)

    def test_straight_motion(self):
        pose=advance(torch.zeros(3),torch.tensor([1.,0.]))
        torch.testing.assert_close(pose,torch.tensor([.25,0.,0.]))

    def test_identification(self):
        f=TerrainFilter(1,"cpu")
        v=torch.zeros(1,2)
        for _ in range(6):
            command=torch.tensor([[1.1,-1.5]])
            obs=motor(v,command,torch.tensor([.2]))
            f.update(motor,v,command,obs)
            v=obs
        self.assertAlmostEqual(f.prob.sum().item(),1.,places=5)
        self.assertLess(abs(f.mean().item()-.2),.07)

    def test_repeatable_map(self):
        for x,y in zip(scenarios(7,58,"cpu"),scenarios(7,58,"cpu")):
            torch.testing.assert_close(x,y)


if __name__=="__main__":
    unittest.main()
