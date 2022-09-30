# module SCH_PARALLEL

# 앙상블 및 분산컴퓨탕

ENV["DISPLAY"]="localhost:10.0"
using Distributed
addprocs(4)

@everywhere using Agents

@everywhere @agent SchellingAgent GridAgent{2} begin
    mood::Bool
    group::Int
end

# for (name, type) in zip(fieldnames(SchellingAgent),fieldtypes(SchellingAgent))
#     println(name,"::",type)
# end

# properties = Dict(:min_to_be_happy => 3)
# schelling = ABM(SchellingAgent, space; properties)

# schelling2 = ABM(
#     SchellingAgent,
#     space;
#     properties = properties,
#     scheduler = Schedulers.ByProperty(:group),
# )

@everywhere using Random

@everywhere function initialize(;numagents = 320, griddims = (20,20), min_to_be_happy = 3, seed = 125)
    space = GridSpaceSingle(griddims, periodic = true)
    properties = Dict(:min_to_be_happy => min_to_be_happy)
    rng = Random.MersenneTwister(seed)
    model = ABM(
        SchellingAgent,space;
        properties, rng, scheduler = Schedulers.Randomly()
    )
    
    for n in 1:numagents
        agent = SchellingAgent(n,(1,1),false,n < numagents / 2 ? 1 : 2 )
        add_agent_single!(agent,model)
    end

    return model
end

@everywhere function agent_step!(agent, model)
    minhappy = model.min_to_be_happy
    count_neighbors_same_group = 0

    for neighbor in nearby_agents(agent, model)
        if agent.group == neighbor.group
            count_neighbors_same_group += 1
        end
    end

    if count_neighbors_same_group ≥ minhappy
        agent.mood = true
    else
        agent.mood = false
        move_agent_single!(agent, model)
    end
    return
end


@everywhere x(agent) = agent.pos[1]
@everywhere adata = [x, :mood, :group]

using BenchmarkTools

#---
# 비분산컴퓨팅
#---

models = [initialize(seed=x) for x in rand(UInt8,3)];
# 총데이터 row : 앙상블수(3) * 에이전트수(320) * 스텝수(5+1 : 0 스텝 포함) = 5,760
@btime agent_df, = ensemblerun!(models, agent_step!, dummystep, 5; adata, parallel=false)
agent_df

# 결과
# 5760×6 DataFrame
#   Row │ step   id     x      mood  group  ensemble 
#       │ Int64  Int64  Int64  Bool  Int64  Int64    
# ──────┼────────────────────────────────────────────
#     1 │     0      1     17  true      1         1
#     2 │     0      2     18  true      1         1
#     3 │     0      3      6  true      1         1
#     4 │     0      4     11  true      1         1
#   ⋮   │   ⋮      ⋮      ⋮     ⋮      ⋮       ⋮
#  5757 │     5    317      8  true      2         3
#  5758 │     5    318     10  true      2         3
#  5759 │     5    319     13  true      2         3
#  5760 │     5    320     14  true      2         3

#---
# 분산컴퓨팅
# ensemblerun!에 parallel=true 추가
#---
models = [initialize(seed=x) for x in rand(UInt8,3)];
@btime agent_df, = ensemblerun!(models,agent_step!,dummystep,5; adata, parallel=true)
agent_df

# end # module
nothing
# fieldnames(SCH.SchellingAgent)
# fieldtypes(SCH.SchellingAgent)
# supertype(SCH.SchellingAgent) : Agents.AbstractAgent
# fieldnames(SCH.GridAgent) : id,:pos
# fieldtypes(SCH.GridAgent) : (Int64, Tuple{Vararg{Int64, D}} where D)
