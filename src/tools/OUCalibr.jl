

using StatsBase

struct OUParams
    kappa::AbstractFloat
    theta::AbstractFloat
    vol  ::AbstractFloat
end


function calcOU(input::Vector{AbstractFloat})
    N   = size(input)[1]
    mu  = mean(input)
    var = variation(input)
    ro  = - log(autocor(input, [1])[1]) 
    vol = sqrt(2.0 * var * ro)

    return OUParams(ro, mu, vol)
end

