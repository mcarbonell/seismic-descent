def test_gd_dt():
    import numpy as np
    x = 600.0
    dt_small = 0.01
    x_small = x
    for _ in range(600):
        x_small -= dt_small * (x_small / 2000.0)
    
    dt_large = 200.0
    x_large = x
    for _ in range(600):
        x_large -= dt_large * (x_large / 2000.0)
        
    print("x_small after 600 steps:", x_small)
    print("x_large after 600 steps:", x_large)

if __name__ == '__main__':
    test_gd_dt()
