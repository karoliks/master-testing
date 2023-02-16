using Z3: Z3
using Z3: Context, real_const, Solver,add, check, get_model,consts
using PyCall
using Conda
#Conda.pip_interop(true; [env::Environment=ROOTENV])
#Conda.pip("install", "z3-solver")
pyimport_conda("z3-solver", "z3-solver")
z3 = pyimport("z3-solver")
math = pyimport("math")
println(math.sin(math.pi / 4)) # returns ≈ 1/√2 = 0.70710678...
ctx = Context()

n = 2
m = 3
v = ones(n, m) #values for the items for the agents

x = real_const(ctx, "x")
y = real_const(ctx, "y")

#A = bit_vec(ctx, "A", n*m)
#D = bit_vec(ctx, "D", n*n*m)

s = Solver(ctx, "QF_NRA")
add(s, x == y^2)
add(s, x > 1)

res = check(s)
@assert res == Z3.sat

m = get_model(s)

for (k, v) in consts(m)
    println("$k = $v")
end
