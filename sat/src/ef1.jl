using PyCall,UnPack
z3=pyimport("z3")
@unpack Or, Not, Bool, 
Solver = z3

Tie= Bool("Tie")
Shirt = Bool("Shirt")
s = Solver()
s.add(Not(Tie))
println(s.check())
println(s.model())