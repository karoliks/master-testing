from z3 import *


s = Solver()

IntSeqSort = SeqSort(IntSort())
SeqSeqSort = SeqSort(IntSeqSort)

sumArray = RecFunction('sumArray', IntSeqSort, IntSort())
sumArrayArg = FreshConst(IntSeqSort)

RecAddDefinition(sumArray, [sumArrayArg], If(Length(sumArrayArg) == 0, 0, sumArrayArg[0] + sumArray(SubSeq(sumArrayArg, 1, Length(sumArrayArg) - 1))
                                             )
                 )


def avgArray(arr):
    return ToReal(sumArray(arr)) / ToReal(Length(arr))


# declare a sequence of integers
seq = Const('seq', IntSeqSort)
allocation = Const('allocation', IntSeqSort)
seqseq = Const("seqseq", SeqSeqSort)


x = Int("x")

# assert the sequence have at least 5 elements
# s.add(Length(seq) >= 5)
# s.add(Length(seq) < x)

dummyIndex = FreshInt('dummyIndex')
# Use numbers between 0 and 10
s.add(ForAll(dummyIndex, Implies(dummyIndex < 9,
      And(seq[dummyIndex] < 10, seq[dummyIndex] > 0))))

dummyIndex2 = FreshInt('dummyIndex2')
s.add(ForAll(dummyIndex2, Implies(dummyIndex2 < 9,
      And(allocation[dummyIndex2] < 3, allocation[dummyIndex2] > 0))))

dummyIndex3 = FreshInt('dummyIndex3')
s.add(ForAll(dummyIndex3, Implies(dummyIndex2 < 9,
      And(seqseq[0][dummyIndex2] < 3, seqseq[0][dummyIndex2] > 0))))

# Have the allocation be boolean
# s.add(ForAll(dummyIndex, Implies(dummyIndex < 9,
#       Or(allocation[dummyIndex] == 0, allocation[dummyIndex] == 1))))
s.add(allocation[1] == 1)

s.add(Length(seq) == x)

s.add(Length(seqseq[0]) == 9)
s.add(Length(seqseq[2]) == 9)

# have the allocation be as big as the seq
s.add(Length(allocation) == x)
# s.add(avgArray(seq) == x)

# get a model and print it:
print(s.check())
if s.check() == sat:
    print("seq", s.model()[seq])
    print("allocation", s.model()[allocation])
    print("seqseq", s.model()[seqseq])
    print
    # counter = 0
    # while s.check() == sat and counter < 15:
    #     counter = counter+1
    #     print("seq", s.model()[seq])
    #     print("allocation", s.model()[allocation])
    #     print
    #     # prevent next model from using the same assignment as a previous model
    #     s.add(Or(x != s.model()[x], seq != s.model()[seq]))
