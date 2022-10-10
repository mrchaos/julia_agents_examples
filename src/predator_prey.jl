using Agents
using Random

@agent Animal GridAgent{2} begin
    energy::Float64
    reproduction_prob::Float64
    Δenergy::Float64    
end

@agent Sheep Animal begin end
@agent Wolf Animal begin end

function initialize_model(;
    n_sheep = 100,
    n_wolves = 50,
    dims = (50, 50),
    regrowth_time = 30,
    Δenergy_sheep = 4,
    Δenergy_wolf = 20,
    sheep_reproduce = 0.04,
    wolf_reproduce = 0.05,
    seed = 23182
)
    rng = MersenneTwister(seed)
    space = GridSpace(dims, periodic = true)

    ## 모델 속성으로 풀이 완전히 다자란 풀과 다시 자라는 걸리는 시간을 나타내는 배열 2개를 포함하고
    ## statici parameter로 풀의 재생시간을 나타내는 `regrowth_time`을 포함한다.
    ## 속성은 `NamedTuple`로 나타낸다.

    properties = (
        fully_grown = falses(dims),
        countdown = zeros(Int, dims),
        regrowth_time = regrowth_time,
    )

    # 모델 생성
    model = ABM(Union{Sheep, Wolf}, space;
        properties, rng, scheduler = Schedulers.randomly, warn = false)

    # 모델에 agent 추가
    for _ in 1:n_sheep
        # 초기 에너지를 음식에서 획득하는 에너지의 2배를 에너지로 설정 한다.          
        energy = rand(model.rng, 1:(Δenergy_sheep*2)) - 1
        add_agent!(Sheep, model, energy, sheep_reproduce, Δenergy_sheep)
    end

    for _ in :1:n_wolves
        energy = rand(model.rng, 1:(Δenergy_wolf*2)) - 1
        add_agent!(Wolf, model, energy, wolf_reproduce, Δenergy_wolf)
    end

    # 모델에 grass추가
    # 모델의 모든 grid좌표에 풀의 속성을 저장한다.
    for p in positions(model)
        fully_grown = rand(model.rng, Bool)
        # 풀이 다자란 경우 풀이 다시 자라기 시작하는 시간을 regrowth_time 이 후 부터
        # 풀이 자라고 있는 경우 0 ~ regrowth_time-1 사이의 지연 시간을 설정
        countdown = fully_grown ? regrowth_time : rand(model.rng, 1:regrowth_time) - 1
        #  예) p = (2,3)
        model.countdown[p...] = countdown
        model.fully_grown[p...] = fully_grown
    end
    return model
end

function sheepwolf_step!(sheep::Sheep, model)
    ## 무작위로 근처로 이동
    walk!(sheep, rand, model) # 내장 함수
    # 에너지 1소비
    sheep.energy -= 1
    # 에너지가 0보다 작으면 양은 죽는다.
    if sheep.energy < 0
        kill_agent!(sheep, model) # 내장 함수
        return
    end
    ## 이동한 지역에 먹을것이 있는 경우 먹는다.
    eat!(sheep, model)
    ## 재생산 확률로 번식한다.
    if rand(model.rng) ≤ sheep.reproduction_prob
        reproduce!(sheep, model)
    end
end

function sheepwolf_step!(wolf::Wolf, model)
    walk!(wolf, rand, model)
    wolf.energy -= 1
    if wolf.energy < 0
        kill_agent!(wolf,model)
        return
    end
    ## 옮긴 지역에 양이 있는 경우 잡아 먹는다.
    dinner = first_sheep_in_position(wolf.pos, model)
    !isnothing(dinner) && eat!(wolf, dinner, model)
    ## 재생산
    if rand(model.rng) ≤ wolf.reproduction_prob
        reproduce!(wolf, model)
    end
end

function first_sheep_in_position(pos, model)
    # 선택된 grid에 있는 양과 늑대의 id 목록을 가지고 온다.
    ids = ids_in_position(pos, model)
    # 가져온 id 목록에서 첫번째 양의 index를 가져온다.
    j = findfirst(id -> model[id] isa Sheep, ids)
    # id 목록에서 첫번째 양의 index에 해당하는 양의 id를 찾아서
    # 해당 양을 가지고 온다.
    if !isnothing(j)
        println("Sheep : ",model[ids[j]].pos, "Wolf : ", pos)
    end
    
    isnothing(j) ? nothing : model[ids[j]]::Sheep
end

function eat!(sheep::Sheep, model)
    # 다자란 풀만 먹는다
    if model.fully_grown[sheep.pos...]
        # 풀먹고 에너지 획득
        sheep.energy += sheep.Δenergy
        # 풀이 있었던자리에 풀을 먹을 수 없음을 표시
        model.fully_grown[sheep.pos...] = false
    end
    return
end

function eat!(wolf::Wolf, sheep::Sheep, model)
    # 늑대 먹이가 된 양은 죽음으로 처리
    kill_agent!(sheep, model)
    # 늑대가 추가적인 에너지를 얻는다.
    wolf.energy += wolf.Δenergy
    return
end

function reproduce!(agent::A, model) where {A}
    agent.energy /= 2
    offspring = deepcopy(agent)
    offspring.id = nextid(model)    
    add_agent_pos!(offspring,model)
    return
end

function grass_step!(model)
    @inbounds for p in positions(model) # @inbounds: 배열의 경계 체크를 끊다
        # 풀이 다 자라지 않은 경우
        if !(model.fully_grown[p...])
            # 풀이 다 성장한 경우 성장완료 표시, countdown에 재성장 시간을 다시 설정 한다.
            if model.countdown[p...] ≤ 0
                model.fully_grown[p...] = true
                model.countdown[p...] = model.regrowth_time
            # 풀이 다 성장 하지 않은 경우 countdown을 차감 하여 성장하고 있음을 나타냄
            else
                model.countdown[p...] -= 1
            end
        end
    end
end

model = initialize_model(n_sheep = 200,n_wolves = 50,dims = (10, 10),)

adf, mdf = run!(model,sheepwolf_step!,grass_step!,2)
