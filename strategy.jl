using LinearAlgebra
using Memoization
using SparseArrays
include("./reputation.jl")
include("./utils.jl")

# AA chance of coop based on its strategy

function AA_coop(stratAA::Int, reputationToJudge::Float64)::Float64
    if (stratAA == utils.STRAT_ALLC) # allC
        return 1.0
    elseif (stratAA == utils.STRAT_ALLD) # allD 
        return 0.0
    else    # disc
        return reputationToJudge 
    end
end

# Average donations of each strategy

function allc_donates(execError::Float64)::Float64
    return (1.0 - execError)
end

function alld_donates()::Float64
    return 0.0
end

function disc_donates_only_human(fAllC::Float64, fAllD::Float64, fDisc::Float64, interactionsAA::Float64, stratAA::Int, execError::Float64, sns::Vector, popsize::Int, gossipRounds::Int)::Float64
    rAllC_H, rAllD_H, rDisc_H, _, _, _, _ = avg_rep(fAllC, fAllD, fDisc, interactionsAA, stratAA, sns, popsize, gossipRounds)

    r = fAllC * rAllC_H + fAllD * rAllD_H + fDisc * rDisc_H      # Disc donates when it interacts with a good human
    return (1.0 - execError) * r
end


function disc_donates_only_AA(fAllC::Float64, fAllD::Float64, fDisc::Float64, interactionsAA::Float64, stratAA::Int, execError::Float64, sns::Vector, popsize::Int, gossipRounds::Int)::Float64
    _, _, _ , _, _, _, rAA_H = avg_rep(fAllC, fAllD, fDisc, interactionsAA, stratAA, sns, popsize, gossipRounds)

    return (1.0 - execError) * rAA_H # Disc donates when it interacts with a good robot
end

function disc_donates(fAllC::Float64, fAllD::Float64, fDisc::Float64, interactionsAA::Float64, stratAA::Int, execError::Float64, sns::Vector, popsize::Int, gossipRounds::Int)::Float64
    # Disc donates when it interacts with a good human or with a good robot
    return (1.0-interactionsAA) * disc_donates_only_human(fAllC, fAllD, fDisc, interactionsAA, stratAA, execError, sns, popsize, gossipRounds) + interactionsAA * disc_donates_only_AA(fAllC, fAllD, fDisc, interactionsAA, stratAA, execError, sns, popsize, gossipRounds)
end

function AA_donates(fAllC::Float64, fAllD::Float64, fDisc::Float64, interactionsAA::Float64, stratAA::Int, execError::Float64, sns::Vector, popsize::Int, gossipRounds::Int)::Float64
    _, _, _, rAllC_AA, rAllD_AA, rDisc_AA, _ = avg_rep(fAllC, fAllD, fDisc, interactionsAA, stratAA, sns, popsize, gossipRounds)

    if (stratAA == utils.STRAT_ALLC) # AllC
        return allc_donates(execError)
    elseif (stratAA == utils.STRAT_ALLD) # AllD
        return alld_donates()
    else # disc
        r = fAllC * rAllC_AA + fAllD * rAllD_AA + fDisc * rDisc_AA

        return (1.0 - execError) * r 
    end
end

# Average receivings of each strategy

function human_receives(rep_H::Float64, rep_AA::Float64, fAllC::Float64, fAllD::Float64, fDisc::Float64, interactionsAA::Float64, stratAA::Int, execError::Float64)::Float64
    # a human receives a donation if:
        #There are no errors:
            #And it interacts with a human and it gets donated by all AllC + all discs that consider it good
            #It interacts with a AA and it gets donated depending on the strategy of the AA:
                #AllC = always, AllD = 1, Disc depends on what the AA thinks of AllCs (rAllC_AA)
    return (1.0 - execError) * ( (1.0 - interactionsAA) * (fAllC + fDisc * rep_H) + interactionsAA * AA_coop(stratAA, rep_AA))
end

function allc_receives(fAllC::Float64, fAllD::Float64, fDisc::Float64, interactionsAA::Float64, stratAA::Int, execError::Float64, sns::Vector, popsize::Int, gossipRounds::Int)::Float64
    rAllC_H, _, _, rAllC_AA, _, _, _ = avg_rep(fAllC, fAllD, fDisc, interactionsAA, stratAA, sns, popsize, gossipRounds)

    return human_receives(rAllC_H, rAllC_AA, fAllC, fAllD, fDisc, interactionsAA, stratAA, execError)
end

function alld_receives(fAllC::Float64, fAllD::Float64, fDisc::Float64, interactionsAA::Float64, stratAA::Int, execError::Float64, sns::Vector, popsize::Int, gossipRounds::Int)::Float64
    _, rAllD_H, _, _, rAllD_AA, _, _ = avg_rep(fAllC, fAllD, fDisc, interactionsAA, stratAA, sns, popsize, gossipRounds)

    return human_receives(rAllD_H, rAllD_AA, fAllC, fAllD, fDisc, interactionsAA, stratAA, execError)
end

function disc_receives(fAllC::Float64, fAllD::Float64, fDisc::Float64, interactionsAA::Float64, stratAA::Int, execError::Float64, sns::Vector, popsize::Int, gossipRounds::Int)::Float64
    _, _, rDisc_H, _, _, rDisc_AA, _ = avg_rep(fAllC, fAllD, fDisc, interactionsAA, stratAA, sns, popsize, gossipRounds)

    return human_receives(rDisc_H, rDisc_AA, fAllC, fAllD, fDisc, interactionsAA, stratAA, execError)
end

function AA_receives(fAllC::Float64, fAllD::Float64, fDisc::Float64, interactionsAA::Float64, stratAA::Int, execError::Float64, sns::Vector, popsize::Int, gossipRounds::Int)::Float64
    _, _, _, _, _, _, rAA_H = avg_rep(fAllC, fAllD, fDisc, interactionsAA, stratAA, sns, popsize, gossipRounds)

    return (1.0 - execError) * (fAllC + fDisc * rAA_H)
end

# Average fitness functions
# These are slightly different from the original paper due to the lack of payoff received from execution errors

function fit_allc(fAllC::Float64, fAllD::Float64, fDisc::Float64, interactionsAA::Float64, stratAA::Int, execError::Float64, b::Float64, sns::Vector, popsize::Int, gossipRounds::Int)::Float64
    
    return b * allc_receives(fAllC, fAllD, fDisc, interactionsAA, stratAA, execError, sns, popsize, gossipRounds) - 1.0 * allc_donates(execError)   # where c = 1, thus -1
end

function fit_alld(fAllC::Float64, fAllD::Float64, fDisc::Float64, interactionsAA::Float64, stratAA::Int, execError::Float64, b::Float64, sns::Vector, popsize::Int, gossipRounds::Int)::Float64

    return b * alld_receives(fAllC, fAllD, fDisc, interactionsAA, stratAA, execError, sns, popsize, gossipRounds) - 1.0 * alld_donates()   # where c = 1, thus -1
end

function fit_disc(fAllC::Float64, fAllD::Float64, fDisc::Float64, interactionsAA::Float64, stratAA::Int, execError::Float64, b::Float64, sns::Vector, popsize::Int, gossipRounds::Int)::Float64
    
    return b * disc_receives(fAllC, fAllD, fDisc, interactionsAA, stratAA, execError, sns, popsize, gossipRounds) - 1.0 * disc_donates(fAllC, fAllD, fDisc, interactionsAA, stratAA, execError, sns, popsize, gossipRounds)   # where c = 1, thus -1
end

function fit_AA(fAllC::Float64, fAllD::Float64, fDisc::Float64, interactionsAA::Float64, stratAA::Int, execError::Float64, b::Float64, sns::Vector, popsize::Int, gossipRounds::Int)::Float64
    
    return b * AA_receives(fAllC, fAllD, fDisc, interactionsAA, stratAA, execError, sns, popsize, gossipRounds) - 1.0 * AA_donates(fAllC, fAllD, fDisc, interactionsAA, stratAA, execError, sns, popsize, gossipRounds)   # where c = 1, thus -1
end


function p_imit(fImitator::Float64, fRoleModel::Float64, sos::Float64=1.0)::Float64
    # Fermi update function, receives fitness of imitator and role model, and strength of selection and returns prob of imitating
    return (1 + exp(-sos * (fRoleModel - fImitator))) ^ -1
end

function damp_AA(strategyToAdopt::Int, stratAA::Int, interactionsAA::Float64)::Float64
    if strategyToAdopt == stratAA return interactionsAA
    elseif return 0.0 end
end

function all_fitness(n_allC::Int, n_allD::Int, sns::Vector, interactionsAA::Float64, stratAA::Int, popsize::Int, execError::Float64, b::Float64, gossipRounds::Int)::Tuple
    # Calculates all the fitness of the given state
    f_allc = fit_allc(n_allC/popsize, n_allD/popsize, (popsize-n_allC-n_allD)/popsize, interactionsAA, stratAA, execError, b, sns, popsize, gossipRounds)
    f_alld = fit_alld(n_allC/popsize, n_allD/popsize, (popsize-n_allC-n_allD)/popsize, interactionsAA, stratAA, execError, b, sns, popsize, gossipRounds)
    f_disc = fit_disc(n_allC/popsize, n_allD/popsize, (popsize-n_allC-n_allD)/popsize, interactionsAA, stratAA, execError, b, sns, popsize, gossipRounds)
    f_AA  = fit_AA(n_allC/popsize, n_allD/popsize, (popsize-n_allC-n_allD)/popsize, interactionsAA, stratAA, execError, b, sns, popsize, gossipRounds)
    return (f_allc, f_alld, f_disc, f_AA)
end

@memoize function transition_prob(fit_imitator::Float64, n_imitator::Int, fit_newstrat::Float64, n_newstrat::Int, stratImit::Int, fitn_AA::Float64, mutChance::Float64, popsize::Int, interactionsAA::Float64, stratAA::Int, imitAAs::Bool)::Float64

    if imitAAs
        return (1 - mutChance) * (n_imitator / popsize) * (( 1 - interactionsAA ) * (n_newstrat / (popsize - 1)) * p_imit(fit_imitator, fit_newstrat) + damp_AA(stratImit, stratAA, interactionsAA) * p_imit(fit_imitator, fitn_AA)) + mutChance * n_imitator / (2 * popsize)
    else
        return (1 - mutChance) * (n_imitator / popsize) * (n_newstrat / (popsize - 1)) * p_imit(fit_imitator, fit_newstrat) + mutChance * n_imitator / (2 * popsize)
    end
end

function grad_disc_alld(n_allC::Int, n_allD::Int, sns::Vector, interactionsAA::Float64, stratAA::Int, popsize::Int, execError::Float64, b::Float64, gossipRounds::Int, mutChance::Float64, imitAAs::Bool)::Float64
    _, fit_alld, fit_disc, fit_AA = all_fitness(n_allC, n_allD, sns, interactionsAA, stratAA, popsize, execError, b, gossipRounds)

    return transition_prob(fit_alld, n_allD, fit_disc, popsize-n_allC-n_allD, utils.STRAT_DISC, fit_AA, mutChance, popsize, interactionsAA, stratAA, imitAAs) - transition_prob(fit_disc, popsize-n_allC-n_allD, fit_alld, n_allD, utils.STRAT_ALLD, fit_AA, mutChance, popsize, interactionsAA, stratAA, imitAAs)
end

@memoize function get_states_strat(popsize::Int)::Vector{Tuple{Int, Int, Int}}
    return [(nAllC, nAllD, popsize - nAllC - nAllD) for nAllC in 0:popsize for nAllD in 0:(popsize - nAllC) if nAllC + nAllD <= popsize]
end

@memoize function stationary_dist_strategy(sns::Vector, interactionsAA::Float64, stratAA::Int, popsize::Int, execError::Float64, b::Float64, mutChance::Float64, gossipRounds::Int, imitAAs::Bool)::Vector{Float64}
    states::Vector{Tuple{Int, Int, Int}} = get_states_strat(popsize)
    lookup = utils.create_lookup_table(states)

    transition_matrix = spzeros(length(states),length(states))

    currentPos = 0
    t1, t2, t3, t4, t5, t6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    for i in 0:popsize
        for j in 0:(popsize - i)
            k = popsize - j - i
            t1, t2, t3, t4, t5, t6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            if k < 0
                continue
            end

            currentPos = utils.pos_of_state(lookup, (i, j, k))

            f_allc, f_alld, f_disc, f_AA = all_fitness(i, j, sns, interactionsAA, stratAA, popsize, execError, b, gossipRounds)

            if i < popsize && j > 0
                transition_matrix[currentPos, utils.pos_of_state(lookup, (i+1, j-1, k))] = t1 = transition_prob(f_alld, j, f_allc, i, utils.STRAT_ALLC, f_AA, mutChance, popsize, interactionsAA, stratAA, imitAAs)
            end
            if i < popsize && k > 0
                transition_matrix[currentPos, utils.pos_of_state(lookup, (i+1, j, k-1))] = t2 = transition_prob(f_disc, k, f_allc, i, utils.STRAT_ALLC, f_AA, mutChance, popsize, interactionsAA, stratAA, imitAAs)
            end
            if i > 0 && j < popsize
                transition_matrix[currentPos, utils.pos_of_state(lookup, (i-1, j+1, k))] = t3 = transition_prob(f_allc, i, f_alld, j, utils.STRAT_ALLD, f_AA, mutChance, popsize, interactionsAA, stratAA, imitAAs)
            end
            if i > 0 && k < popsize
                transition_matrix[currentPos, utils.pos_of_state(lookup, (i-1, j, k+1))] = t4 = transition_prob(f_allc, i, f_disc, k, utils.STRAT_DISC, f_AA, mutChance, popsize, interactionsAA, stratAA, imitAAs)
            end
            if k > 0 && j < popsize
                transition_matrix[currentPos, utils.pos_of_state(lookup, (i, j+1, k-1))] = t5 = transition_prob(f_disc, k, f_alld, j, utils.STRAT_ALLD, f_AA, mutChance, popsize, interactionsAA, stratAA, imitAAs)
            end
            if j > 0 && k < popsize
                transition_matrix[currentPos, utils.pos_of_state(lookup, (i, j-1, k+1))] = t6 = transition_prob(f_alld, j, f_disc, k, utils.STRAT_DISC, f_AA, mutChance, popsize, interactionsAA, stratAA, imitAAs)
            end
            transition_matrix[currentPos,currentPos] = 1 - t1 - t2 - t3 - t4 - t5 - t6
        end
    end

    result = utils.get_transition_matrix_statdist(transition_matrix)

    return result
end

function stat_dist_at_point_strat(n_allC::Int, n_allD::Int, sns::Vector, interactionsAA::Float64, stratAA::Int, popsize::Int, execError::Float64, b::Float64, mutChance::Float64, gossipRounds::Int, imitAAs::Bool)::Float64
    states = get_states_strat(popsize)

    if (n_allC, n_allD, popsize-n_allC-n_allD) ∉ states return 0 end

    stat_dist = stationary_dist_strategy(sns, interactionsAA, stratAA, popsize, execError, b, mutChance, gossipRounds, imitAAs)
    pos = utils.find_pos_of_state(states, (n_allC, n_allD, popsize-n_allC-n_allD))
    result = stat_dist[pos]

    return result
end

function gradient_of_selection_at_state(allC::Int, allD::Int, disc::Int, sns::Vector, interactionsAA::Float64, stratAA::Int, popsize::Int, execError::Float64, b::Float64, mutChance::Float64, gossipRounds::Int, imitAAs::Bool)::Tuple{Float64,Float64,Float64}
    # For a given state, calculates the gradient of selection at that state
    grad = [0.0, 0.0, 0.0]

    f_allc, f_alld, f_disc, f_AA = all_fitness(allC, allD, sns, interactionsAA, stratAA, popsize, execError, b, gossipRounds)
    # allC = prob(+AllC) - prob(-AllC)
    grad[1] = (transition_prob(f_alld, allD, f_allc, allC, utils.STRAT_ALLC, f_AA, mutChance, popsize, interactionsAA, stratAA, imitAAs) + transition_prob(f_disc, disc, f_allc, allC, utils.STRAT_ALLC, f_AA, mutChance, popsize, interactionsAA, stratAA, imitAAs)) -
            (transition_prob(f_allc, allC, f_alld, allD, utils.STRAT_ALLD, f_AA, mutChance, popsize, interactionsAA, stratAA, imitAAs) + transition_prob(f_allc, allC, f_disc, disc, utils.STRAT_DISC, f_AA, mutChance, popsize, interactionsAA, stratAA, imitAAs))

    grad[2] = (transition_prob(f_allc, allC, f_alld, allD, utils.STRAT_ALLD, f_AA, mutChance, popsize, interactionsAA, stratAA, imitAAs) + transition_prob(f_disc, disc, f_alld, allD, utils.STRAT_ALLD, f_AA, mutChance, popsize, interactionsAA, stratAA, imitAAs)) -
            (transition_prob(f_alld, allD, f_allc, allC, utils.STRAT_ALLC, f_AA, mutChance, popsize, interactionsAA, stratAA, imitAAs) + transition_prob(f_alld, allD, f_disc, disc, utils.STRAT_DISC, f_AA, mutChance, popsize, interactionsAA, stratAA, imitAAs))

    grad[3] = (transition_prob(f_allc, allC, f_disc, disc, utils.STRAT_DISC, f_AA, mutChance, popsize, interactionsAA, stratAA, imitAAs) + transition_prob(f_alld, allD, f_disc, disc, utils.STRAT_DISC, f_AA, mutChance, popsize, interactionsAA, stratAA, imitAAs)) -
            (transition_prob(f_disc, disc, f_allc, allC, utils.STRAT_ALLC, f_AA, mutChance, popsize, interactionsAA, stratAA, imitAAs) + transition_prob(f_disc, disc, f_alld, allD, utils.STRAT_ALLD, f_AA, mutChance, popsize, interactionsAA, stratAA, imitAAs))

    return (grad[1], grad[2], grad[3])
end

function gradient_of_selection(sns::Vector, interactionsAA::Float64, stratAA::Int, popsize::Int, execError::Float64, b::Float64, mutChance::Float64, gossipRounds::Int, imitAAs::Bool)::Vector
    # returns the gradient of selection at each state
    states = get_states_strat(popsize)
    all_grads = []
    for state in states
        grad = gradient_of_selection_at_state(state[1], state[2], state[3], sns, interactionsAA, stratAA, popsize, execError, b, mutChance, gossipRounds, imitAAs)
        push!(all_grads, grad)
    end
    return all_grads
end

function coop_at_state(allC::Int, allD::Int, disc::Int, sns::Vector, interactionsAA::Float64, stratAA::Int, popsize::Int, execError::Float64, b::Float64, mutChance::Float64, gossipRounds::Int, imitAAs::Bool)::Tuple{Float64,Float64,Float64,Float64,Float64}
    # At each state, we return 5 cooperation indexes:
    # coop_X_Y is the cooperation between X and Y; coop_H is the cooperation of humans including with both humans and AAs; coop_All is the total cooperation index of the system
    fAllC, fAllD, fDisc = allC/popsize, allD/popsize, disc/popsize

    coop_H_H, coop_H_AA, coop_AA_H, coop_H, coop_All = 0.0, 0.0, 0.0, 0.0, 0.0

    aaDonates = AA_donates(fAllC, fAllD, fDisc, interactionsAA, stratAA, execError, sns, popsize, gossipRounds)
    allcDonates = allc_donates(execError)
    alldDonates = alld_donates() 
    discDonatesH = disc_donates_only_human(fAllC, fAllD, fDisc, interactionsAA, stratAA, execError, sns, popsize, gossipRounds)
    discDonatesAA = disc_donates_only_AA(fAllC, fAllD, fDisc, interactionsAA, stratAA, execError, sns, popsize, gossipRounds)

    coop_H_H = allcDonates * fAllC + alldDonates * fAllD + discDonatesH * fDisc
    coop_H_AA = allcDonates * fAllC + alldDonates * fAllD + discDonatesAA * fDisc
    coop_AA_H = aaDonates
    coop_H = coop_H_H * (1-interactionsAA) + coop_H_AA * interactionsAA
    coop_All = coop_H * (1-interactionsAA) + coop_AA_H * interactionsAA

    return (coop_H_H, coop_H_AA, coop_AA_H, coop_H, coop_All)
end

function coop_index(sns::Vector, interactionsAA::Float64, stratAA::Int, popsize::Int, execError::Float64, b::Float64, mutChance::Float64, gossipRounds::Int, imitAAs::Bool=true)::Tuple
    # returns the 5 cooperation indexes averaged by statDist, plus all the cooperation indexes at each state
    states = get_states_strat(popsize)
    coop_indexes = [0.0, 0.0, 0.0, 0.0, 0.0]
    all_coop_index = []
    stationaryDist = 0.0

    for state in states
        stationaryDist = stat_dist_at_point_strat(state[1], state[2], sns, interactionsAA, stratAA, popsize, execError, b, mutChance, gossipRounds, imitAAs)
        coopIndex = coop_at_state(state[1], state[2], state[3], sns, interactionsAA, stratAA, popsize, execError, b, mutChance, gossipRounds, imitAAs)
        push!(all_coop_index, coopIndex)
        coop_indexes .+= stationaryDist .* coopIndex
    end
    return (coop_indexes, all_coop_index)
end

function avg_reputations(sns::Vector, interactionsAA::Float64, stratAA::Int, popsize::Int, execError::Float64, b::Float64, mutChance::Float64, gossipRounds::Int, imitAAs::Bool=true)::Tuple
    # returns the 7 resulting reputations averaged by statDist, plus all the reputations at each state
    states = get_states_strat(popsize)
    avg_r = [Float64(0),Float64(0),Float64(0),Float64(0),Float64(0),Float64(0), Float64(0)]
    all_r = []
    stationaryDist = 0.0

    for state in states
        stationaryDist = stat_dist_at_point_strat(state[1], state[2], sns, interactionsAA, stratAA, popsize, execError, b, mutChance, gossipRounds, imitAAs)
        rep = avg_rep(state[1]/popsize, state[2]/popsize, state[3]/popsize, interactionsAA, stratAA, sns, popsize, gossipRounds)
        push!(all_r, rep)
        avg_r .+= stationaryDist .* rep
    end
    return (avg_r, all_r)
end

function avg_agreement(sns::Vector, interactionsAA::Float64, stratAA::Int, popsize::Int, execError::Float64, b::Float64, mutChance::Float64, gossipRounds::Int, imitAAs::Bool)::Tuple
    # returns the 4 agreement and disagreements averaged by statDist, plus all the agreement and disagreements at each state
    states = get_states_strat(popsize)
    avg_ag = [Float64(0),Float64(0),Float64(0),Float64(0)]
    all_ag = []
    stationaryDist = 0.0

    for state in states
        stationaryDist = stat_dist_at_point_strat(state[1], state[2], sns, interactionsAA, stratAA, popsize, execError, b, mutChance, gossipRounds, imitAAs)
        ag = agreement_and_disagreement(state[1]/popsize, state[2]/popsize, state[3]/popsize, interactionsAA, stratAA, sns, popsize, gossipRounds)
        push!(all_ag, ag)
        avg_ag .+= stationaryDist .* ag
    end
    return (avg_ag, all_ag)
end

function avg_state(sns::Vector, interactionsAA::Float64, stratAA::Int, popsize::Int, execError::Float64, b::Float64, mutChance::Float64, gossipRounds::Int, imitAAs::Bool=true)::Tuple{Float64,Float64,Float64}
    # returns the average state of the system
    states = get_states_strat(popsize)
    avg_s = [Float64(0),Float64(0),Float64(0)]
    stationaryDist = 0.0

    for state in states
        stationaryDist = stat_dist_at_point_strat(state[1], state[2], sns, interactionsAA, stratAA, popsize, execError, b, mutChance, gossipRounds, imitAAs)
        avg_s .+= stationaryDist .* state
    end
    return (avg_s[1], avg_s[2], avg_s[3])
end

function get_all_data(sns::Vector, interactionsAA::Float64, stratAA::Int, popsize::Int, execError::Float64, b::Float64, mutChance::Float64, gossipRounds::Int, imitAAs::Bool=true)::Tuple
    # Returns all possible data from the model. coopIndex, reputations, statDist, avgState, agreement, gradOfSel.
    coop = coop_index(sns, interactionsAA, stratAA, popsize, execError, b, mutChance, gossipRounds, imitAAs)
    rep = avg_reputations(sns, interactionsAA, stratAA, popsize, execError, b, mutChance, gossipRounds, imitAAs)
    statDist = stationary_dist_strategy(sns, interactionsAA, stratAA, popsize, execError, b, mutChance, gossipRounds, imitAAs)
    avgState = avg_state(sns, interactionsAA, stratAA, popsize, execError, b, mutChance, gossipRounds, imitAAs)
    agreement = avg_agreement(sns, interactionsAA, stratAA, popsize, execError, b, mutChance, gossipRounds, imitAAs)
    grad = gradient_of_selection(sns, interactionsAA, stratAA, popsize, execError, b, mutChance, gossipRounds, imitAAs)
    return (coop, rep, statDist, avgState, agreement, grad)
end

# --------------------------
# Run results
# --------------------------

function vary_variable(params::Vector, parameterVaried::String, indexParamToVary::Int, rangeOfValues::Vector, tag::String, foldername::String="", addErrorToNorm::Bool=false) 
    # Given a foldername, a list of parameters, and the parameter to vary within a given interval, calculates all data and stores it

    folder_path = utils.make_plot_folder(foldername)

    sns, interactionsAA, stratAA, popsize, execError, assessError, b, mutChance, gossipRounds, strengthOfSelection, imitAAs = params

    
    utils.write_parameters(folder_path, sns, interactionsAA, stratAA, popsize, execError, assessError, b, mutChance, gossipRounds, strengthOfSelection,parameterVaried, rangeOfValues, imitAAs)
    
    # Create a list with all results
    array_of_any = [nothing for _ in 1:length(rangeOfValues)]
    allResults = Vector{Any}(array_of_any)
    
    for i in 1:length(rangeOfValues)
        
        # change our parameter acording to range provided
        params[indexParamToVary] = rangeOfValues[i]
        sns, interactionsAA, stratAA, popsize, execError, assessError, b, mutChance, gossipRounds, _ = params
        if (addErrorToNorm)
            sns = [add_errors_sn(norm, execError, assessError) for norm in sns]
        end

        #println("Running simulations for $tag: $parameterVaried = ",  rangeOfValues[i], ". ", i,"/",length(rangeOfValues))
        allResults[i] = get_all_data(sns, interactionsAA, stratAA, popsize, execError, b, mutChance, gossipRounds, imitAAs)
    
        Memoization.empty_all_caches!()
    end

    utils.write_all_results(folder_path, tag, allResults)
    println("Simulation done! Saving results.")


    println("Results saved.")
end