using Z3
using Plots
using Graphs 
using GraphRecipes
using CSV

# ctx = Context()
# x = real_const(ctx, "x")
# y = real_const(ctx, "y")

# s = Solver(ctx, "QF_NRA")
# add(s, x == y^2)
# add(s, x > 1)

# res = check(s)
# @assert res == Z3.sat

# m = get_model(s)

# for (k, v) in consts(m)
#     println("$k = $v")
# end



function get_formula_for_one_item_to_one_agent(A, n, m, ctx)
    formulas = []
    # Each item allocated to at exactly one agent
    for g in 1:m
        push!(formulas, sum([ite(A[i][g], int_val(ctx,1), int_val(ctx,0)) for i in 1:n]) == 1)
    end
    return and([formula for formula in formulas])
end




function get_mms_for_this_agent(this_agent, n, m, V, G)
    # Z3.setOption("pp-decimal", true)
    formulas = []
    ctx = Context()
    
    # println("hei2")
    # TODO hvilket tall? og vanlig variabel eller z3 variabel?
    mms = real_const(ctx,"mms_"* string(this_agent))

    # A  keeps track of the allocated items
    A = [[bool_const(ctx,"a_"*string(i)*"_"*string(j)*"_mms_calculation_"*string(this_agent)) for j in 1:m]  # TODO må jeg ha denne? virker som det blir mye vfor z3 å tenke på
         for i in 1:n]
    # println("hei3")

    for i in 1:n
        push!(formulas,mms <= sum(
            #todo endre pga column major?
            [ite(A[i][g], real_val(ctx,V[this_agent,g]), real_val(ctx,0)) for g in 1:m]))  # look at this_agents values, because this is from her point of view
    end
    opt = Optimize(ctx)
    set(ctx,"timeout", 300000)  # TODO increase timeout

    # set(opt,"timeout", 5000)  # TODO increase timeout
    add(opt,and(formulas))
    add(opt,get_formula_for_one_item_to_one_agent(A, n, m, ctx))
    # opt.add(get_edge_conflicts(G, A, n)) # TODO take back to test
    maximize(opt,mms)
    println(check(opt) == Z3.sat)
    mod = get_model(opt)
    ans = -1.0
    # println(mod)

    if check(opt) == Z3.sat
        for (k, v) in consts(mod)
            if string(k) == string(mms)
                # println(typeof(Z3_get_numeral_decimal_string(v)))
                # println("isreal: ",is_real(v))
                println("$k = $((get_decimal_string(v, 3)))" )
                ans = parse(Float64,get_decimal_string(v, 3))#parse(Float64, string(eval(v))) # TODO better conversion
                # ans = real_val(ctx,v)
            end
        end
    else
        return false
    end

    # println(ans)

    return ans
end

# TODO ikke bruke maksimering her? 
function maximin_shares(n, m, V, G)

    individual_mms = []
    
    for i in 1:n
        # println("mms i: ", i, " n: ", n)
        push!(individual_mms,get_mms_for_this_agent(i, n, m, V, G))
        # println(typeof(individual_mms[i]))
        # if typeof(individual_mms[i]) != Z3.ExprAllocated 
        #     return false
        # end
    end    
    
    ctx = Context()
    alpha_mms_agents = [real_const(ctx,"alpha_agent_$(string(i+1))") for i in 1:n]

    # A  keeps track of the allocated items
    A = [[bool_const(ctx,"a_$(i+1)_$(j+1)") for j in 1:m]
         for i in 1:n]

    for i in 1:n
        # println(typeof(individual_mms[i] > real_val(ctx,0)))
        alpha_mms_agents[i] = ite(bool_val(ctx,(individual_mms[i] > 0)), sum([ite(
            A[i][g], real_val(ctx,V[i,g]), real_val(ctx,0)) for g in 1:m]) / individual_mms[i], real_val(ctx,1))
    end

    opt = Optimize(ctx)

    for i in 1:n
        bundle_value = sum(
           [ite(A[i][g], real_val(ctx,V[i,g]), real_val(ctx,0)) for g in 1:m])
        # opt.add_soft(individual_mms[i] <= bundle_value)
        maximize(opt, bundle_value)
        # TODO maximere bundle-verdiene i tillegg?
    end

    add(opt,get_formula_for_one_item_to_one_agent(A, n, m, ctx))
    # opt.add(get_edge_conflicts(G, A, n)) # TODO take back to test

    # println(check(opt))
    mod = get_model(opt)
   
    for (k, v) in consts(mod)
        println("$k = $(eval(v))")
    end

    return check(opt) == Z3.sat
end



function erdos_renyi_experiment()

    times = Float64[]
    agents = Float64[]
    items = Float64[]

    timed_out_counter = 0
    discarded_graph_counter = 0

    for i in 1:100
        n = rand(2:10)
        m = rand(n*2:n*3)
        p = rand()

        V = rand(0:100, n, m)
        # print(V)
        graph = Graphs.erdos_renyi(m, p)
        # graphplot(graph)
        # Plots.savefig("Erdos_Renyi.png")
        # max_degree = maximum(Graphs.degree(graph))
        println("i:", i, "  n:", n, " m:", m)#, "    max deg:", max_degree)

        # if max_degree >= n
        #     discarded_graph_counter = discarded_graph_counter + 1
        #     continue
        # end

        # TODO rather use benchmarkttols? time is used now besasuse the function may be very slow

        elapsed_time = @elapsed maximin_shares(n, m, V, graph)

      
        println("elapsed_time: ", elapsed_time)

        push!(times, elapsed_time)
        push!(agents, n)
        push!(items, m)
    end

    CSV.write("mms_no_conflicts_z3_julia.csv", (times = times, agents=agents, items=items))

    println("timed_out_counter", timed_out_counter)
    println("discarded_graph_counter", discarded_graph_counter)

    # plotting the points
    plot(agents, times, seriestype=:scatter, label="Z3")

    # naming the x axis
    xlabel!("agents")
    # naming the y axis
    ylabel!("execution time (seconds)")

    Plots.savefig("no_conflicts_mms_z3_agents_julia_4a.pdf")

    # function to show the plot
    # Plots.show()

    # plotting the points
    plot(items, times, seriestype=:scatter, label="Z3")

    # naming the x axis
    xlabel!("items")
    # naming the y axis
    ylabel!("execution time (seconds)")

    # giving a title to my graph
    Plots.savefig("erdos_renyi_mms_z3_items_julia_4a.pdf")

    # function to show the plot
    # Plots.show()

end


erdos_renyi_experiment()