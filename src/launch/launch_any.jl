using DataFrames

function set_envs(c::Dict)
    df = CSV.read(c["path_df"], DataFrame)
    
    train_env = Env(
        df[c["train_start_idx"] : c["train_end_idx"], :],
        "BTCUSDT_FutT";
        w_size       = c["trading_step"], 
        commission   = c["commission"], 
        exclude_cols = c["exclude_cols"],
        need_to_scale= c["need_to_scale"]
    )
    test_env = Env(
        df[c["test_start_idx"] : c["test_end_idx"], :],
        "BTCUSDT_FutT"; 
        w_size       = c["trading_step"], 
        commission   = c["commission"], 
        exclude_cols = c["exclude_cols"],
        need_to_scale = c["need_to_scale"],
        scaler = train_env.scaler
    )

    return train_env, test_env

end

function set_mode_actions(c::Dict, discr::Bool)
    mode = @match c["mode"] begin
        "spread" => spread
        "ou"     => OU
        "as"     => AS
    end

    stat_algo = nothing
    actions::Any = 0

    if mode == AS
        stat_algo = init_as!(c)
        if (discr)
            actions   = [Iterators.product(T.(c["as_alpha_vars"]), 
            T.(c["as_k_vars"]), T.(c["as_gamma_vars"]))...]
        else
            actions = 3
        end
    elseif mode == OU
        stat_algo = init_mr!(c)
        if (discr)
            actions   = [Iterators.product(T.(
                c["mr_in_k_vars"]), T.(c["mr_out_k_vars"]), T.(c["mr_crit_k_vars"]))...]
        else
            actions   = 3
        end
    elseif mode == spread
        if (discr)
            action_base = T.(c["actions"]) ./ 10000.0
            actions     = [Iterators.product(action_base, action_base)...]
            pushfirst!(actions, (0.0, 0.0))
        else
            actions     = 2
        end
    end

    return stat_algo, actions, mode

end

function set_model(
    c::Dict, 
    train_env::Env, 
    actions::Union{Int, Vector{Any}},
    stat_algo::Union{Nothing, StatAlgo},
    mode::ActionType
)
    model = @match c["type"] begin
        "dqn"   => init!(
                    in_feats     = length(train_env.feats_for_model),
                    out_feats    = length(actions),
                    layers       = c["layers"],
                    action_space = actions,
                    stat_algo    = stat_algo,
                    action_type  = mode
                )
        "ddpg"  => init_ddpg!(
                    in_feats = length(train_env.feats_for_model),
                    A_layers = c["A_layers"],
                    C_layers = c["C_layers"],
                    action_space = actions,
                    action_type = mode,
                    stat_algo = stat_algo
                )
        "td3"   => init_td3!(
                    in_feats = length(train_env.feats_for_model),
                    A_layers = c["A_layers"],
                    C_layers = c["C_layers"],
                    action_space = actions,
                    action_type = mode,
                    stat_algo = stat_algo
                )
    end

    return model
end

function launch(c::Dict)
    train_env, test_env = set_envs(c)

    discr_act = c["type"] == "dqn"
    stat_algo, actions, mode = set_mode_actions(c, discr_act)
    model = set_model(c, train_env, actions, stat_algo, mode)

    eval_res = train!(c, model, train_env, test_env, c["eval_save_path_pref"])

    res = Dict("eval" => eval_res, "model" => model, "config" => c)
    return res
end

