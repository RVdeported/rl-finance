using Pkg
Pkg.activate("./alp")
Pkg.instantiate()
using YAML, Plots, Base.Threads, Wandb
include("../tools/BaseTypes.jl")
include("../tools/Env.jl")
include("../agents/MRbase.jl")
include("../agents/AS.jl")
include("../agents/Grid.jl")
include("../agents/DQN.jl")
include("../agents/DDPG.jl")
include("../agents/TD3.jl")
include("../agents/PPO.jl")
include("../launch/launch_any.jl")


SUMMARY_PATH = "./experiments/exp-$(now()).jld2"

CONFIGS = [
    # "./config/dqn_alpha.yml",
    # "./config/dqn_alpha.yml",
    # "./config/dqn_alpha.yml",
    # "./config/ddpg_alpha.yml",
    # "./config/ddpg_alpha.yml",
    # "./config/ddpg_alpha.yml",
    # "./config/td3_alpha.yml",
    # "./config/td3_alpha.yml",
    # "./config/td3_alpha.yml",
    # "./config/dqn_ou_alpha.yml",
    # "./config/dqn_ou_alpha.yml",
    # "./config/dqn_ou_alpha.yml",
    # "./config/ddpg_ou_alpha.yml",
    # "./config/ddpg_ou_alpha.yml",
    # "./config/ddpg_ou_alpha.yml",
    # "./config/td3_ou_alpha.yml",
    # "./config/td3_ou_alpha.yml",
    # "./config/td3_ou_alpha.yml",
    # "./config/dqn_as_alpha.yml",
    # "./config/dqn_as_alpha.yml",
    # "./config/dqn_as_alpha.yml",
    # "./config/ddpg_as_alpha.yml",
    # "./config/ddpg_as_alpha.yml",
    # "./config/ddpg_as_alpha.yml",
    # "./config/td3_as_alpha.yml",
    # "./config/td3_as_alpha.yml",
    # "./config/td3_as_alpha.yml",
    # "./config/dqn_gr_alpha.yml",
    # "./config/dqn_gr_alpha.yml",
    # "./config/dqn_gr_alpha.yml",
    # "./config/ddpg_gr_alpha.yml",
    # "./config/ddpg_gr_alpha.yml",
    # "./config/ddpg_gr_alpha.yml",
    # "./config/td3_gr_alpha.yml",
    # "./config/td3_gr_alpha.yml",
    # "./config/td3_gr_alpha.yml",
    "./config/ppo_alpha.yml",
    "./config/ppo_ou_alpha.yml",
    "./config/ppo_as_alpha.yml",
    "./config/ppo_gr_alpha.yml",
]

lg = WandbLogger(
    project = "rl-finance-alpha",
    name    = "alpha-$(now())",
    config  = Dict("configs" => length(CONFIGS))
)
# lg = nothing

println("Using $(nthreads()) threads")
res_vec = Any[ nothing for _ in CONFIGS ]
def_as = CONFIGS[3]
def_ou = CONFIGS[2]
def_as_conf = YAML.load_file(def_as)
def_ou_conf = YAML.load_file(def_ou)
push!(res_vec, Dict(
    "model"  => init_as!(def_as_conf),
    "config" => def_as_conf
))
push!(res_vec, Dict(
    "model"  => init_mr!(def_ou_conf),
    "config" => def_ou_conf   
))


#------------------------start of loop---------------#
# try

@threads for id in eachindex(CONFIGS)
    println("Starting $(CONFIGS[id])...")
    c = YAML.load_file(CONFIGS[id])
    
    c["path_df"]         = "/home/ugrek/Maquette/data/diploma/BTCUSDT_features_8192.csv"
    c["episodes"]        = 1000
    c["train_start_idx"] = 20000
    c["train_end_idx"]   = 3400000
    c["warm_up_episodes"]= 100

    res_vec[id] = launch(c, lg)
end

push!(CONFIGS, def_as)
push!(CONFIGS, def_ou)
eval_pnl    = AbstractFloat[-Inf for _ in res_vec]
eval_sharpe = AbstractFloat[-Inf for _ in res_vec]
eval_rew    = [[] for _ in res_vec]

@threads for i in eachindex(res_vec)
    model = res_vec[i]["model"]
    c = res_vec[i]["config"]
    c["test_start_idx"] = 3500000
    c["test_end_idx"]   = 5150000

    (model isa RLModel) && move_pr(model, true)
    _, test_env = set_envs(c)

    eval_r = simulate_live!(
        deepcopy(test_env),
        model, c, lg
    )
    eval_pnl[i] = eval_r.PnL
    eval_rew[i] = eval_r.reward
    eval_sharpe[i] = sharpe(eval_r.reward)
    (model isa RLModel) && move_pr(model, false)
end

# catch err
#     println(err)
#     if lg isa WandbLogger
#         close(lg)
#     end
# end
#------------------------end of loop---------------#

for i in eachindex(CONFIGS)
    println("$(CONFIGS[i])\t$(eval_pnl[i])\t$(eval_sharpe[i])")
end

out = Dict(
    "configs" => CONFIGS,
    "pnl"     => eval_pnl,
    "models"  => res_vec,
    "stats"   => eval_rew,
    "sharpe"  => eval_sharpe
)

@save "$SUMMARY_PATH" out