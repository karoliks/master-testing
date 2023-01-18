using Z3: Z3
using Z3: Context, real_const, Solver,add, check, get_model,consts
ctx = Context()
x = real_const(ctx, "x")
y = real_const(ctx, "y")

s = Solver(ctx, "QF_NRA")
add(s, x == y^2)
add(s, x > 1)

res = check(s)
@assert res == Z3.sat

m = get_model(s)

for (k, v) in consts(m)
    println("$k = $v")
end