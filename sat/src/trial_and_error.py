from z3 import *


def more_than_100(x):
    return And(x > 100)


def less_than_100(x):
    return And(x < 100)


def simple_logic():
    s = Solver()

    x = Int("x")
    v = Int("v")

    s.add(ForAll(x, And(x*v != 1, x != 1)))

    print(s.check())
    if s.check() == sat:
        print(s.model())


def simple_logic_2():
    s = Solver()

    x = Int("x")
    v = Int("v")

    # Kommer ikke med fordi den kan være hva som helst?
    # I den første implikasjonen sier vi ikke at noe ikke kan skje, vi bare understreker at et spesialtilfelle er ok, men forsåvidt bør da alle andre spesialtilfeller også være ok da.
    # s.add(Implies(x == v, x == 7))
    # s.add(Implies(x == 5, False))
    # s.add(x == v)
    s.add(ForAll(x, x == 8))

    print(s.check())
    if s.check() == sat:
        counter = 0
        while s.check() == sat and counter < 15:
            counter = counter+1
            print(s.model())
            # prevent next model from using the same assignment as a previous model
            s.add(Or(x != s.model()[x], v != s.model()[v]))
        # print(s.model())


if __name__ == "__main__":
    simple_logic()
    simple_logic_2()
