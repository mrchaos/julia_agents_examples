using Agents
using LinearAlgebra
using Random

@agent Bird ContinuousAgent{2} begin
    speed::Float64
    cohere_factor::Float64
    separation::Float64
    separate_factor::Float64
    match_factor::Float64
    visual_distance::Float64
end
fieldnames(Bird)

function initialize_model(
    n_birds = 100,  # bird 수
    speed = 1.0,# 1 step당 속력 (방향은 vel에 정의된곳)
    cohere_factor = 0.25, # 이웃 새들의 평균 위치를 유지하는 중요성
    separation = 4.0, # 이웃 새들과 떨어져 있어야 하는 최소 거리
    separate_factor = 0.25, # 이웃 새들과의 최소거리를 유지하는 중요성
    match_factor = 0.01, #  이웃 새들의 평균 궤적 일치의 중요성
    visual_distance = 5.0, # 새가 볼 수 있는 거리를 나타내며 이웃 새의 반경을 정의
    extent = (100,100),
)
    # ContinuousSpace(extent;spacing = minimum(extent)/20.0,update_vel! = no_vel_update,periodic = true)
    space2d = ContinuousSpace(extent;spacing= visual_distance/1.5)

    # AgentBasedModel(Bird,space::S = nothing; scheduler::F = Schedulers.fastest, 
    #    properties::P = nothing, rng::R = Random.default_rng(), warn = true)
    model = ABM(Bird, space2d, scheduler = Schedulers.Randomly())

    for _ in 1:n_birds
        vel = Tuple(rand(model.rng, 2)*2 .- 1) #  -1 < vel < 1
        add_agent!(model,
            vel,
            speed,
            cohere_factor,
            separation,
            separate_factor,
            match_factor,
            visual_distance,            
        )
    end
    return model
end


function agent_step!(bird, model)
    ## 새의 가시거리(visual distance)에 있는 이웃 새들의 id 획득
    neighbor_ids = nearby_ids(bird, model, bird.visual_distance)
    N = 0 # 이웃한 새들의 수
    match = separate = cohere = (0.0, 0.0)
    
    # 이웃한 새들에 기반한 행위 속성(behaiviour properties) 계산
    for id in neighbor_ids
        N += 1
        # model[id]는 model.agents[1] 과 동일하며 Bird를 돌려 준다.
        neighbor = model.agents[id].pos
        # 이웃새를 향한 위치 vector , 현재 선택된 새를 기준으로 다른 새들의 상대적 위치 vector
        heading = neighbor .- bird.pos
        
        ## `cohere` : 이웃한 새들의 평균 위치를 계산 : 규치2를 위한 계산
        cohere = cohere .+ heading # 위치벡터 합 계산
        
        # 규칙 1
        # 이웃 새와 떨어져 있어야할 최소 거리보다 작은 경우 이웃 새로 부터 멀어 진다. 
        if euclidean_distance(bird.pos, neighbor, model) < bird.separate
            # 이웃한 새를 향한 반대 방향으로 상대적 위치 만큼 멀어 진다.
            # 떨어지는 거리의 합산 계산
            separate = separate .- heading
        end
        ## `match` : 이웃한 새들의 평균 궤적(average trajectory)을 계산
        # 이웃한 새들의 속도를 합산 
        # ※ 속도 : 속력과 방량이 있음
        match = match .+ model.agents[id].vel
    end
    #  최소한 이웃한 새가 1마리는 있어야 함, 아래 계산상 N이 0이 되면 안됨
    N = max(N,1)
    
    ## 모델 입력값과 이웃새들의 수에 기반한 결과들을 normalize
    ## 가중치의 역할 : 새의 속도를 계산할 때 규칙1,2,3의 중요도에 대한 가중치
    ##  즉 Separation에 가중치가 크면 새들이 서로 회피하는 방향으로 많이 움직임
    ##  Cohesion의 가중치가 크면 새들이 서로 뭉치는 경향이 큼
    ##  Alignment의 가중치가 크면 새들이 서로 같은 방향으로 움직이는 경향이 큼 
    
    ## 규칙2 (응집)
    # 이웃새들과의 평균 위치에 가중치를 곱함    
    cohere = cohere ./ N .* bird.cohere_factor
    
    ## 규칙1 (분산)
    separate = separate ./ N .* bird.separate_factor
    
    ##규칙3 (정렬)
    match = match ./ N .* bird.match_factor
    
    # 위에 정의된 규칙에 기반하여 속도(velocity)를 계산
    # cohere, separate,match 는 1step(시간)당 움직는 거리 이므로 속도 개념
    # 기존 속도와 현재 계산된 속도의 평균값이므로 2로 나눔
    bird.vel = (bird.vel .+ cohere .+ separate .+ match) ./ 2
    #  속도 normalize
    bird.vel = bird.vel ./ norm(bird.vel)
    
    ## 새로 계산된 속도와 속력으로 움직이기
    move_agent!(bird, model, bird.speed)
end

using InteractiveDynamics
using CairoMakie
CairoMakie.activate!()

const bird_polygon = Polygon(Point2f[(-0.5,-0.5),(1,0),(-0.5,0.5)]);

function bird_marker(b::Bird)
    # 새의 방향 :  tan(ϕ) = vy/vx = vel[2] / vel[1], ϕ = atan(vel[2]/vel[x])    
    ϕ = atan(b.vel[2], b.vel[1]) # + π/2 + π
    scale(rotate2D(bird_polygon,ϕ), 2)
end