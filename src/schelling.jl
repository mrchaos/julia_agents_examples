# module SCH
# GLMakie를 사용하고 싶은 경우 설정에 대해서는 https://julialang.kr/?p=3684 를 참조
ENV["DISPLAY"]="localhost:12.0"

using Agents

# space = GridSpaceSingle((10,10); periodic = false)

@agent SchellingAgent GridAgent{2} begin
    mood::Bool
    group::Int
end

fieldnames(GridAgent{2})

GridAgent{2}

for (name, type) in zip(fieldnames(SchellingAgent),fieldtypes(SchellingAgent))
    println(name,"::",type)
end

# properties = Dict(:min_to_be_happy => 3)
# schelling = ABM(SchellingAgent, space; properties)

# schelling2 = ABM(
#     SchellingAgent,
#     space;
#     properties = properties,
#     scheduler = Schedulers.ByProperty(:group),
# )

using Random

function initialize(;numagents = 320, griddims = (20,20), min_to_be_happy = 3, seed = 125)
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

function agent_step!(agent, model)
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

# model = initialize()

# step!(model,agent_step!)

# step!(model, agent_step!, 3)



using InteractiveDynamics
# using CairoMakie
# CairoMakie.activate!()

using GLMakie
GLMakie.activate!()

# model = initialize(; numagents = 300,seed=now().instant.periods.value)
model = initialize(; numagents = 300)

groupcolor(a) = a.group == 1 ? :blue : :orange
groupmarker(a) = a.group == 1 ? :circle : :rect
params = Dict(
    :min_to_be_happy => 1:10,
)
plotkwargs =(;ac = groupcolor, am = groupmarker, as = 10)

#--------------------------
# plot
#--------------------------

fig, ax, abmobs = abmplot(model;
    agent_step! = agent_step!, model_step! = dummystep,
    params, plotkwargs...)

fig


#--------------------------
# video
#--------------------------
# model = initialize()

# abmvideo(
#     "schelling.mp4",
#     model, agent_step!, dummystep;
#     title = "Schelling", frames = 20,
#     framerate = 1,
#     plotkwargs...
# )

#--------------------------
# Collecting data
#--------------------------

using Statistics: mean

#---
# agent 구조체의 필드에 대해 수집
#---
adata = [:pos, :mood, :group]
model = initialize(;numagents=3, )
# n=10 : 모든 agent에 대해 0 ~ 10 step까지 진행시 pos, mood, group 데이터를 수집한다.
# 수집데이터 row는 agent# x step(n>=0) =  3 x 11 (0 step 포함) = 33 
# run!의 리턴은 (agent_dataframe, model_dataframe)
agent_df, model_df = run!(model, agent_step!,10;adata)
agent_df[1:30,:]

#---
# agent 구조체와 function 조합의 데이터 수집
#---
# adata의 원소로 들어가는 function의 파라미터는 agent임
# pos : (x,y)
x(agent) = agent.pos[1]
adata = [x, :mood, :group]
model = initialize(;numagents=3,min_to_be_happy=2)

agent_df, model_df = run!(model, agent_step!, 10; adata)
agent_df[1:30,:]

#---
# agent 구조체와 function 조합의 데이터 수집 및 통계 데이터
#---
adata = [(:mood,sum),(x, mean)]
agent_df, model_df = run!(model,agent_step!,10;adata)
agent_df

#---
# agent 구조체와 function 조합의 데이터 수집 및 통계 그래프
#---
model = initialize()
params = Dict(:min_to_be_happy => 0:8)
alabels = ["happy","avg. x"]
fig, abmobs = abmexploration(model;
    agent_step! , model_step! = dummystep, params, plotkwargs...,
    adata, alabels, 
)
fig


#--------------------------
# Save / Load
#--------------------------
@eval Main __atexample_named_schelling = $(@__MODULE__)

# agent 200개가 400step후에 안정화 된것으로 기대
model = initialize(numagents=200, min_to_be_happy=4, seed=42)
run!(model, agent_step!, 400)

fig, ax, abmobs = abmplot(model; plotkwargs...)
fig

#---
# save model : HDF5 기반 데이터 포맷으로 저장됨
#---
AgentsIO.save_checkpoint("schelling.jld2",model)

model = AgentsIO.load_checkpoint("schelling.jld2"; scheduler = Schedulers.Randomly())

#---
# 추가로 100개의 agent 투입 후 40 step 진행
#---
for i in 1:100
    agent = SchellingAgent(nextid(model),(1,1),false,1)
    add_agent_single!(agent,model)
end
fig, ax, abmobs = abmplot(model; plotkwargs...)
fig

run!(model,agent_step!,40)
fig, ax, abmobs = abmplot(model; plotkwargs...)
fig

#---
# 저장된 모델 로딩 후 새로운 그룹 추가
#---
# 모델로딩
model = AgentsIO.load_checkpoint("schelling.jld2"; scheduler=Schedulers.Randomly())

# 새로운 그룹 추가
for i in 1:100
    agent = SchellingAgent(nextid(model),(1,1),false,3)
    add_agent_single!(agent,model)
end

# group color redefine
groupcolor(a) = (:blue, :orange, :green)[a.group]
groupmarker(a) = (:circle, :rect, :cross)[a.group]
#plotkwargs =(;ac = groupcolor, am = groupmarker, as = 10)

fig, ax, abmobs = abmplot(model; plotkwargs...)
fig

run!(model,agent_step!,40)
fig, ax, abmobs = abmplot(model; plotkwargs...)
fig

rm("schelling.jld2")

#---
# Scanning parameter ranges
#  - ABM의 동작에 대해 다양한 매개변수의 효과에 관심이 있는 경우
#  - parameters에 들어가는 dict의 값이 Vector타입이면 expand된다.
#    아래의 경우 ":min_to_be_happy"와 ":numagents"는 Vector타입 이고
#    ":griddims"는 Tuple타입으로 ":min_to_be_happy"와 ":numagents"의 Vector size의
#    곱과 스텝(0스텝포함)의 곱만큼 데이터 row가 생긴다.
#    4*2*(3+1)  = 32
#
#   - paramscan function은 병렬처리 가능하다.
#     parallet=true로 설정하면 됨
#    
#---
happyperc(moods) = count(moods) / length(moods)
adata = [(:mood, happyperc)]
parameters = Dict(
    :min_to_be_happy => collect(2:5),  # expanded
    :numagents => [200,300], # expanded
    :griddims => (20,20), # not vector = not expanded
)
agent_df, model_df = paramscan(parameters, initialize; adata, agent_step!,n=3)
agent_df
# end # module
nothing
# fieldnames(SCH.SchellingAgent)
# fieldtypes(SCH.SchellingAgent)
# supertype(SCH.SchellingAgent) : Agents.AbstractAgent
# fieldnames(SCH.GridAgent) : id,:pos
# fieldtypes(SCH.GridAgent) : (Int64, Tuple{Vararg{Int64, D}} where D)
