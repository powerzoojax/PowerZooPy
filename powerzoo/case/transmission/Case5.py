from powerzoo.case.CaseBase import ClearCase, DataFrame


class Case5(ClearCase):
    GRID_TYPE = "transmission"
    BUS_COUNT = 5
    VOLTAGE_LEVEL = "HV"
    SOURCE = "MATPOWER"
    DESCRIPTION = "IEEE 5-bus test system"

    def __init__(self, *args, **kwargs):
        # type: 1=PQ (load), 2=PV (gen), 3=Ref (θ=0 in DCOPF)
        # Pd/Qd: nominal load at each bus (MW / MVAr)
        self.nodes = DataFrame(
            ['id', 'type', 'Pd', 'Qd', 'x', 'y'],
            [[1.0, 3,   0.0,   0.0, 0, 0],     # Reference bus (2 gens)
             [2.0, 1, 300.0,  98.6, 3, 0],     # Load bus
             [3.0, 2, 300.0,  98.6, 4.5, 1],   # Gen bus
             [4.0, 2, 400.0, 131.5, 2, 2],     # Gen bus
             [5.0, 2,   0.0,   0.0, 0, 2]])    # Gen bus

        self.units = DataFrame(
            ['id', 'bus_id', 'mc_a', 'mc_b', 'mc_c', 'p_max', 'p_min'],
            [[1.0, 1.0, 0.0, 0.0, 14.0, 40.0, 5.0],
             [2.0, 1.0, 0.0, 0.0, 15.0, 170.0, 10.0],
             [3.0, 3.0, 0.0, 0.0, 30.0, 520.0, 20.0],
             [4.0, 4.0, 0.0, 0.0, 40.0, 200.0, 10.0],
             [5.0, 5.0, 0.0, 0.0, 10.0, 600.0, 20.0]])

        self.lines = DataFrame(
            ['id', 'from', 'to', 'x', 'floor', 'cap'],
            [[1.0, 1.0, 2.0, 0.0281, -400.0, 400.0],
             [2.0, 1.0, 4.0, 0.0304, 0.0, 0.0],
             [3.0, 1.0, 5.0, 0.0064, 0.0, 0.0],
             [4.0, 2.0, 3.0, 0.0108, 0.0, 0.0],
             [5.0, 3.0, 4.0, 0.0297, 0.0, 0.0],
             [6.0, 4.0, 5.0, 0.0297, -240.0, 240.0]])

        self.loads = DataFrame(
            ['id', 'bus_id', 'mc_a', 'mc_b', 'mc_c', 'd_max', 'd_min'],
            [[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [2.0, 2.0, 0.0, 0.0, 0.0, 500.0, 0.0],
             [3.0, 3.0, 0.0, 0.0, 0.0, 600.0, 0.0],
             [4.0, 4.0, 0.0, 0.0, 0.0, 400.0, 0.0],
             [5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        
        self.real_params = True
        super().__init__(*args, **kwargs)

if __name__ == '__main__':
    c = Case5()
    c.check()
    print(c.get_node_ptdf())
