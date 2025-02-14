using LinearAlgebra
using NonlinearSolve 
using Memoization

# Add errors (assign and execute) to social norm
function add_errors_sn(sn::Vector, execError::Float64, assessmentError::Float64)::Vector
    newsn = [0.0,0.0,0.0,0.0]

    p = sn[3]
    q = sn[4]

    newsn[1] = (1 - execError) * (1 - assessmentError) + execError * assessmentError
    newsn[2] = assessmentError
    newsn[3] = p * (newsn[1] - assessmentError) + q * (1 - newsn[1] - assessmentError) + assessmentError
    newsn[4] = q * (1 - 2 * assessmentError) + assessmentError

    return newsn
end

# ODEs to determine the average reputation of each strategy

function avg_rep_ode!(dr, r, p)
    fAllC, fAllD, fDisc, interactionsAA, stratAA, sns, popsize, gossipRounds = p
    snH_H_H, snH_AA_H, snH_H_AA, snAA_H_H, snAA_H_AA = sns # X_Y_Z = X observing Y playing against Z

    # The average reputation in an interaction for humans with humans
    rall_H_H = fAllC * r[1] + fAllD * r[2] + fDisc * r[3]
    # The average reputation in an interaction for humans with machines
    rall_H_AA = r[7]
    # The average reputation in an interaction for machines with humans
    rall_AA_H = fAllC * r[4] + fAllD * r[5] + fDisc * r[6]

    # Pre gossip reputation alignment metrics for humans interacting with humans
    # These now have to account for the fraction of time you interact with the machine too
    g2init_H_H = fAllC * r[1]^2             + fAllD * r[2]^2            + fDisc * r[3]^2   # Fraction of agreement that a focal individual is good
    b2init_H_H = fAllC * (1-r[1])^2         + fAllD * (1-r[2])^2        + fDisc * (1-r[3])^2   # Fraction of agreement that a focal individual is bad
    d2init_H_H = fAllC * r[1] * (1-r[1])    + fAllD * r[2] * (1-r[2])   + fDisc * r[3] * (1-r[3])   # Fraction of disagreement about a focal individual

    # Post gossip reputation alignment metrics for humans interacting with humans-> using peer-to-peer
    T = gossipRounds / popsize
    g2_H_H = g2init_H_H + d2init_H_H * (1 - ℯ^(-T))
    b2_H_H = b2init_H_H + d2init_H_H * (1 - ℯ^(-T))
    d2_H_H = d2init_H_H * ℯ^(-T)

    # Pre gossip reputation alignment metrics for humans interacting with machines
    # These now have to account for the fraction of time you interact with the machine too
    g2init_H_AA = r[7]^2
    b2init_H_AA = (1-r[7])^2
    d2init_H_AA = r[7] * (1-r[7])

    # Post gossip reputation alignment metrics for humans interacting with machines-> using peer-to-peer
    g2_H_AA = g2init_H_AA + d2init_H_AA * (1 - ℯ^(-T))
    b2_H_AA = b2init_H_AA + d2init_H_AA * (1 - ℯ^(-T))
    d2_H_AA = d2init_H_AA * ℯ^(-T)

    # no gossip exists between the AA, so the alignment metrics are constant when it observes
    # in this case, we consider 4 scenarios: 
    #   Both AA and H see the recipient as good (g2), both see it as bad (b2)
    #   But now we got to seperate d2 to use the two possibilities for the social norm:
    #       Either the donor, human, sees the recipient as good, and the AA as bad (d2_hg_aab_H_AA)
    #       Or donor sees recipient as bad, and the AA as good (d2_hb_aag_H_AA)
    g2_AA_H =        fAllC * r[1] * r[4]        + fAllD * r[2] * r[5]       + fDisc * r[3] * r[6]
    b2_AA_H =        fAllC * (1-r[1]) * (1-r[4])+ fAllD * (1-r[2]) *(1-r[5])+ fDisc * (1-r[3]) * (1-r[6])
    d2_hg_aab_AA_H = fAllC * r[1] * (1-r[4])    + fAllD * r[2] * (1-r[5])   + fDisc * r[3] * (1-r[6])
    d2_hb_aag_AA_H = fAllC * (1-r[1]) * r[4]    + fAllD * (1-r[2]) * r[5]   + fDisc * (1-r[3]) * r[6]

    # ODE problem for reputations of humans by humans
    # These now have to account for the fraction of time you interact with the machine too
    dr[1] = (1-interactionsAA) * (rall_H_H * snH_H_H[1] + (1 - rall_H_H) * snH_H_H[3]) + interactionsAA * (rall_H_AA * snH_H_AA[1] + (1-rall_H_AA) * snH_H_AA[3]) - r[1]

    dr[2] = (1-interactionsAA) * (rall_H_H * snH_H_H[2] + (1 - rall_H_H) * snH_H_H[4]) + interactionsAA * (rall_H_AA * snH_H_AA[2] + (1-rall_H_AA) * snH_H_AA[4]) - r[2]

    dr[3] = (1-interactionsAA) * (g2_H_H * snH_H_H[1] + d2_H_H * (snH_H_H[3] + snH_H_H[2]) + b2_H_H * snH_H_H[4]) + interactionsAA * (g2_H_AA * snH_H_AA[1] + d2_H_AA * (snH_H_AA[3] + snH_H_AA[2]) + b2_H_AA * snH_H_AA[4]) - r[3]

    # ODE problem for reputations of humans by AA
    # AA only see humans interacting with themselves or with itself. It always thinks of itself as good though.
    dr[4] = (1-interactionsAA) * (rall_AA_H * snAA_H_H[1] + (1 - rall_AA_H) * snAA_H_H[3]) + interactionsAA * (1.0 * snAA_H_AA[1] + (1-1.0) * snAA_H_AA[3]) - r[4]

    dr[5] = (1-interactionsAA) * (rall_AA_H * snAA_H_H[2] + (1 - rall_AA_H) * snAA_H_H[4]) + interactionsAA * (1.0 * snAA_H_AA[2] + (1-1.0) * snAA_H_AA[4]) - r[5]

    dr[6] = (1-interactionsAA) * (g2_AA_H * snAA_H_H[1] + d2_hg_aab_AA_H * snAA_H_H[3] + d2_hb_aag_AA_H * snAA_H_H[2] + b2_AA_H * snAA_H_H[4]) + interactionsAA * (1.0 * r[7] * snAA_H_AA[1] + 1.0 * (1-r[7]) * snAA_H_AA[2]) - r[6]   # AA thinks himself is good, human thinks AA is good/bad. Human either cooperates with good or defects with good

    # ODE problem for reputations of AA by human
    # An AA gets a good reputation following the social norm snH_AA_H
    # Simpler than the above as the AA only interacts with humans
    if (stratAA == utils.STRAT_ALLC) # if AA is allC
        dr[7] = rall_H_H * snH_AA_H[1] + (1 - rall_H_H) * snH_AA_H[3] - r[7]
    elseif (stratAA == utils.STRAT_ALLD) # if AA is allD
        dr[7] = rall_H_H * snH_AA_H[2] + (1 - rall_H_H) * snH_AA_H[4] - r[7]
    else # if AA is Disc. Notice the switched sn 2 and 3, due to the human's perspective
        dr[7] = g2_AA_H * snH_AA_H[1] + d2_hg_aab_AA_H * snH_AA_H[2] + d2_hb_aag_AA_H * snH_AA_H[3] + b2_AA_H * snH_AA_H[4] - r[7]
    end

end

@memoize function avg_rep(fAllC::Float64, fAllD::Float64, fDisc::Float64, interactionsAA::Float64, stratAA::Int, sns::Vector, popsize::Int, gossipRounds::Int)::Tuple{Float64,Float64,Float64,Float64,Float64,Float64, Float64}
    average_rep_strat0 = [0.5,0.5,0.5,0.5,0.5,0.5,0.5]      # initial reputation state (rAllC_H, rAllD_H, rDisc_H, rAllC_AA, rAllD_AA, rDisc_AA, rAA_H)
                                                            # where rX_Y is the reputation of group X in the eyes of Y

    nl_prob = NonlinearProblem(avg_rep_ode!, average_rep_strat0, (fAllC, fAllD, fDisc, interactionsAA, stratAA, sns, popsize, gossipRounds))
    sol = solve(nl_prob,NewtonRaphson())

    return Tuple(clamp(x, 0, 1) for x in sol.u)
end

# Agreement level of private reputations

function agreement_H_H(fAllC::Float64, fAllD::Float64, fDisc::Float64, interactionsAA::Float64, stratAA::Int, sns::Vector, popsize::Int, gossipRounds::Int)::Float64
    rAllC, rAllD, rDisc, _, _, _, _ = avg_rep(fAllC, fAllD, fDisc, interactionsAA, stratAA, sns, popsize, gossipRounds)

    return fAllC * rAllC^2 + fAllD * rAllD^2 + fDisc * rDisc^2 + fAllC * (1 - rAllC)^2 + fAllD * (1 - rAllD)^2 + fDisc * (1 - rDisc)^2
end

function agreement_H_AA(fAllC::Float64, fAllD::Float64, fDisc::Float64, interactionsAA::Float64, stratAA::Int, sns::Vector, popsize::Int, gossipRounds::Int)::Float64
    rAllC_H, rAllD_H, rDisc_H, _, _, _, rAA_H = avg_rep(fAllC, fAllD, fDisc, interactionsAA, stratAA, sns, popsize, gossipRounds)

    return (fAllC * rAllC_H + fAllD * rAllD_H + fDisc * rDisc_H) * rAA_H + (fAllC * (1 - rAllC_H) + fAllD * (1 - rAllD_H) + fDisc * (1 - rDisc_H)) * (1 - rAA_H)
end

function disagreement_H_H(fAllC::Float64, fAllD::Float64, fDisc::Float64, interactionsAA::Float64, stratAA::Int, sns::Vector, popsize::Int, gossipRounds::Int)::Float64
    rAllC, rAllD, rDisc, _, _, _, _ = avg_rep(fAllC, fAllD, fDisc, interactionsAA, stratAA, sns, popsize, gossipRounds)

    return 2 * ( fAllC * rAllC * (1-rAllC)    + fAllD * rAllD * (1-rAllD)   + fDisc * rDisc * (1-rDisc))
end

function disagreement_H_AA(fAllC::Float64, fAllD::Float64, fDisc::Float64, interactionsAA::Float64, stratAA::Int, sns::Vector, popsize::Int, gossipRounds::Int)::Float64
    rAllC_H, rAllD_H, rDisc_H, _, _, _, rAA_H = avg_rep(fAllC, fAllD, fDisc, interactionsAA, stratAA, sns, popsize, gossipRounds)

    return fAllC * rAllC_H * (1-rAA_H)    + fAllD * rAllD_H * (1-rAA_H)   + fDisc * rDisc_H * (1-rAA_H) + fAllC * (1 - rAllC_H) * rAA_H    + fAllD * (1 - rAllD_H) * rAA_H   + fDisc * (1 - rDisc_H) * rAA_H
end

function agreement_and_disagreement(fAllC::Float64, fAllD::Float64, fDisc::Float64, interactionsAA::Float64, stratAA::Int, sns::Vector, popsize::Int, gossipRounds::Int)::Tuple{Float64, Float64, Float64, Float64}
    #returns all the agreements and disagreements
    return (agreement_H_H(fAllC, fAllD, fDisc, interactionsAA, stratAA, sns, popsize, gossipRounds),disagreement_H_H(fAllC, fAllD, fDisc, interactionsAA, stratAA, sns, popsize, gossipRounds),
    agreement_H_AA(fAllC, fAllD, fDisc, interactionsAA, stratAA, sns, popsize, gossipRounds),disagreement_H_AA(fAllC, fAllD, fDisc, interactionsAA, stratAA, sns, popsize, gossipRounds))
end