import robots as r

def test_analytical(target , joints, model):
    print("Result" , model.computeMGD(joints))
    print("Attempt" , target , "\n")